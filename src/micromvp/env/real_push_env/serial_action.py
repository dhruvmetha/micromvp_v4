"""
Serial action sender for real robot control (Aggregated Protocol).

Sends wheel speed commands to multiple robots via a single Serial Port.

Protocol Structure (Aggregated):
-----------------------------------------------------------------------------------------
| Header (2B) | Seq (2B) | Count (1B) | Body (N * 5B)             | Checksum (1B)       |
-----------------------------------------------------------------------------------------
| 0xAA 0x55   | uint16   | uint8      | [ID(1B) L(2B) R(2B)] ...  | Sum(Seq..Body) & FF |
-----------------------------------------------------------------------------------------

- Speed values are in range [-1000, 1000] representing normalized [-1.0, 1.0].
- Robot ID is uint8 (0-255).
"""
from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Union

import serial  # pip install pyserial
from micromvp.core.models import Action


@dataclass
class SerialActionConfig:
    """Configuration for Serial action sender."""
    # 串口配置
    port: str = "/dev/ttyACM0"
    baudrate: int = 115200

    # 发送频率 (Hz)
    send_hz: float = 30.0

    # 是否反转右轮 (部分电机接线相反)
    invert_right_wheel: bool = False

    # 预设的活跃 Robot ID 列表 (用于初始化)
    # 串口是广播总线，如果你只控制 ID 1 和 2，可以在这里指定
    initial_robot_ids: List[int] = field(default_factory=list)


@dataclass
class APStatusInfo:
    """AP status information parsed from serial output."""
    is_connected: bool = False
    alive_robot_ids: List[int] = field(default_factory=list)
    recent_events: List[str] = field(default_factory=list)  # Max 10 events
    last_update_time: float = 0.0


class SerialActionSender:
    """
    Serial-based action sender.
    Replaces the original UDPActionSender while keeping the API compatible.
    """

    def __init__(self, config: SerialActionConfig) -> None:
        self._config = config
        self._running = False

        # 串口对象
        self._ser: Optional[serial.Serial] = None
        self._serial_lock = threading.Lock()  # 保护串口写入，防止多线程冲突

        # 状态管理
        # robot_id -> Action
        self._actions: Dict[int, Action] = {}
        # 追踪活跃的 Robot ID (替代原来的 Endpoint 列表)
        self._active_robot_ids: set = set(config.initial_robot_ids)

        self._action_lock = threading.Lock()  # 保护 _actions 字典

        # 全局序列号 (所有机器人共用一个时间轴)
        self._global_seq: int = 0

        # 后台线程
        self._thread: Optional[threading.Thread] = None

        # AP status monitoring
        self._ap_status = APStatusInfo()
        self._status_lock = threading.Lock()
        self._reader_thread: Optional[threading.Thread] = None
        self._pending_alive_ids: List[int] = []  # Collects IDs until STATUS line
        self._previous_alive_ids: set = set()  # Track previous alive IDs for connect/disconnect events

        # Throttled write-error logging. Without this a hiccup on the
        # ESP32 USB-JTAG path floods the log at the 30 Hz send rate.
        # Strategy: print first occurrence and then a periodic summary
        # ("N timeouts in the last T seconds") so the failure mode is
        # still visible without the spam.
        self._write_err_count: int = 0
        self._write_err_window_start: float = 0.0
        self._write_err_last_print: float = 0.0
        self._write_err_summary_interval_sec: float = 5.0

    def start(self) -> bool:
        """Start the serial connection and sender thread."""
        if self._running:
            return True

        try:
            self._ser = serial.Serial(
                port=self._config.port,
                baudrate=self._config.baudrate,
                timeout=0.1,
                # write_timeout was 0.1s but the ESP32 USB-JTAG path can lag
                # briefly under load and the per-send window is only ~33 ms
                # (30 Hz). 0.5 s gives the OS write buffer plenty of room
                # to drain while still bounding latency so a truly stuck
                # serial doesn't block forever.
                write_timeout=0.5,
            )
            print(f"[SerialSender] Port {self._config.port} opened at {self._config.baudrate}")
        except serial.SerialException as e:
            print(f"[SerialSender] Error opening port: {e}")
            return False

        self._running = True

        # Mark AP as connected
        with self._status_lock:
            self._ap_status.is_connected = True

        # Start sender thread
        self._thread = threading.Thread(target=self._send_loop, daemon=True)
        self._thread.start()

        # Start reader thread for AP status monitoring
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        print(f"[SerialSender] Started at {self._config.send_hz} Hz")
        return True

    def stop(self) -> None:
        """Stop the sender and close serial port."""
        if not self._running:
            return

        self._running = False

        # Stop sender thread
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        # Stop reader thread
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
            self._reader_thread = None

        # 发送全停指令
        self.stop_all()

        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

        # Mark AP as disconnected
        with self._status_lock:
            self._ap_status.is_connected = False

        print("[SerialSender] Stopped")

    def set_action(self, robot_id: int, action: Action) -> None:
        """Set action for a robot (API Compatible)."""
        with self._action_lock:
            self._actions[robot_id] = action
            self._active_robot_ids.add(robot_id)

    def set_actions(self, actions: Dict[int, Action]) -> None:
        """Set actions for multiple robots (API Compatible)."""
        with self._action_lock:
            for robot_id, action in actions.items():
                self._actions[robot_id] = action
                self._active_robot_ids.add(robot_id)

    def send_immediate(self, robot_id: int, action: Action) -> bool:
        """
        Send action immediately (bypassing the loop rate).
        Note: In serial bus, this inserts a packet for this specific robot immediately.
        """
        # 更新缓存状态
        self.set_action(robot_id, action)
        
        # 立即构建并发送单机包
        return self._send_packet({robot_id: action})

    def stop_all(self) -> None:
        """Send stop command to all known robots immediately."""
        stop_actions = {}
        with self._action_lock:
            for robot_id in self._active_robot_ids:
                stop_action = Action.stop()
                self._actions[robot_id] = stop_action
                stop_actions[robot_id] = stop_action
        
        # 立即发送停止帧
        self._send_packet(stop_actions)

    def add_robot(self, robot_id: int, ip: str = "", port: int = 0) -> None:
        """
        Register a new robot.
        Arguments `ip` and `port` are ignored in Serial mode but kept for API compatibility.
        """
        with self._action_lock:
            self._active_robot_ids.add(robot_id)
            if robot_id not in self._actions:
                self._actions[robot_id] = Action.stop()

    def remove_robot(self, robot_id: int) -> None:
        """Unregister a robot."""
        with self._action_lock:
            self._active_robot_ids.discard(robot_id)
            self._actions.pop(robot_id, None)

    def get_ap_status(self) -> APStatusInfo:
        """Get current AP status (thread-safe copy)."""
        with self._status_lock:
            return APStatusInfo(
                is_connected=self._ap_status.is_connected,
                alive_robot_ids=list(self._ap_status.alive_robot_ids),
                recent_events=list(self._ap_status.recent_events),
                last_update_time=self._ap_status.last_update_time,
            )

    # -------------------------------------------------------------------------
    # Internal - AP Status Reader
    # -------------------------------------------------------------------------

    def _reader_loop(self) -> None:
        """Background thread that reads and parses AP status from serial."""
        while self._running:
            if self._ser is None or not self._ser.is_open:
                time.sleep(0.1)
                continue

            try:
                # Read line with short timeout (set in Serial init)
                line_bytes = self._ser.readline()
                if not line_bytes:
                    continue

                line = line_bytes.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                self._parse_status_line(line)

            except serial.SerialException:
                # Serial disconnected or error
                time.sleep(0.1)
            except Exception:
                # Ignore parsing errors
                pass

    def _parse_status_line(self, line: str) -> None:
        """Parse a status line from Arduino AP."""
        if line.startswith("[ALIVE]"):
            # [ALIVE] id=X ip=A.B.C.D last=Ums rxHello=N
            # Extract robot ID and add to pending list
            try:
                # Parse id=X
                parts = line.split()
                for part in parts:
                    if part.startswith("id="):
                        robot_id = int(part[3:])
                        if robot_id not in self._pending_alive_ids:
                            self._pending_alive_ids.append(robot_id)
                        break
            except (ValueError, IndexError):
                pass

        elif line.startswith("[STATUS]"):
            # [STATUS] stations=X hello=N serFrames=M udpTx=P cksumBad=Q drop=R notReg=S
            # This signals end of ALIVE list, update status
            current_ids = set(self._pending_alive_ids)

            # Detect newly connected cars
            new_connections = current_ids - self._previous_alive_ids
            # Detect disconnected cars
            disconnections = self._previous_alive_ids - current_ids

            with self._status_lock:
                # Add connection events (and surface them on stdout so anyone
                # tailing the run log can see live connectivity changes — the
                # GUI sidebar already shows recent_events but the log didn't.)
                for robot_id in sorted(new_connections):
                    msg = f"[SerialSender] Car {robot_id} CONNECTED (alive now: {sorted(current_ids)})"
                    self._add_event(msg)
                    print(msg, flush=True)

                for robot_id in sorted(disconnections):
                    msg = f"[SerialSender] ⚠️ Car {robot_id} DISCONNECTED (alive now: {sorted(current_ids)})"
                    self._add_event(msg)
                    print(msg, flush=True)

                self._ap_status.alive_robot_ids = sorted(self._pending_alive_ids)
                self._ap_status.last_update_time = time.time()

            # Update previous IDs for next comparison
            self._previous_alive_ids = current_ids
            self._pending_alive_ids = []

        elif line.startswith("[WARN]") or line.startswith("[ERR]"):
            # Add to recent events
            with self._status_lock:
                self._add_event(line)

    def _add_event(self, event: str) -> None:
        """Add an event to the recent events list. Must be called with _status_lock held."""
        self._ap_status.recent_events.append(event)
        # Keep only last 10 events
        if len(self._ap_status.recent_events) > 10:
            self._ap_status.recent_events = self._ap_status.recent_events[-10:]

    # -------------------------------------------------------------------------
    # Internal - Protocol Implementation
    # -------------------------------------------------------------------------

    def _send_loop(self) -> None:
        """Background loop."""
        period = 1.0 / max(1.0, self._config.send_hz)

        while self._running:
            loop_start = time.perf_counter()

            # 1. 获取当前所有动作快照
            with self._action_lock:
                # 过滤掉不在 active_list 里的，虽然理论上 set_action 会同步更新
                current_actions = {
                    rid: act for rid, act in self._actions.items() 
                    if rid in self._active_robot_ids
                }

            # 2. 发送聚合包
            if current_actions:
                self._send_packet(current_actions)

            # 3. 控频
            elapsed = time.perf_counter() - loop_start
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _send_packet(self, actions_map: Dict[int, Action]) -> bool:
        """
        Construct and write the aggregated packet to serial.
        Thread-safe wrapper around serial write.
        """
        if self._ser is None or not self._ser.is_open:
            return False

        payload_body = bytearray()
        valid_count = 0

        # 构建 Body
        for robot_id, action in actions_map.items():
            # 限制 RobotID 范围 (0-255)
            if not (0 <= robot_id <= 255):
                continue

            # 速度处理
            left = max(-1.0, min(1.0, action.left_speed))
            right = max(-1.0, min(1.0, action.right_speed))
            
            if self._config.invert_right_wheel:
                right = -right

            l_int = int(left * 1000)
            r_int = int(right * 1000)

            # Pack: [ID (1B)] [Left (2B)] [Right (2B)]
            # <Bhh = Little Endian: uchar, short, short
            payload_body.extend(struct.pack("<Bhh", robot_id, l_int, r_int))
            valid_count += 1

        if valid_count == 0:
            return False

        with self._serial_lock:
            try:
                # 1. 更新序列号
                seq = self._global_seq
                self._global_seq = (seq + 1) & 0xFFFF

                # 2. 构建 Header 部分: [Seq (2B)] [Count (1B)]
                header_info = struct.pack("<HB", seq, valid_count)

                # 3. 计算 Checksum
                # Checksum = Sum(HeaderInfo + Body) & 0xFF
                check_data = header_info + payload_body
                checksum = sum(check_data) & 0xFF

                # 4. 组装完整帧: [Head AA 55] + [CheckData] + [Checksum]
                full_packet = b"\xAA\x55" + check_data + struct.pack("B", checksum)

                # 5. 写入串口
                self._ser.write(full_packet)
                return True
            except serial.SerialException as e:
                # Throttle the error message: print on first failure (so the
                # user sees the failure mode immediately), then suppress until
                # the summary interval elapses; emit one summary line with the
                # aggregate count and clear the window.
                now = time.time()
                self._write_err_count += 1
                if self._write_err_count == 1:
                    print(f"[SerialSender] ⚠️ Serial write error: {e}", flush=True)
                    self._write_err_window_start = now
                    self._write_err_last_print = now
                elif now - self._write_err_last_print >= self._write_err_summary_interval_sec:
                    window = now - self._write_err_window_start
                    print(
                        f"[SerialSender] ⚠️ {self._write_err_count} write errors "
                        f"in last {window:.1f}s (most recent: {e})",
                        flush=True,
                    )
                    self._write_err_count = 0
                    self._write_err_window_start = now
                    self._write_err_last_print = now
                return False
            except Exception as e:
                print(f"Unexpected serial error: {e}")
                return False