import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import PortAudioBackend as portaudio
import RealtimeAudioEngine as audio_engine


def _write_wav(path: Path, fs_hz: int, values: np.ndarray) -> None:
    samples = np.asarray(np.clip(values, -1.0, 1.0) * 32767.0, dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(fs_hz)
        handle.writeframes(samples.tobytes())


class _CallbackStop(Exception):
    pass


class _CallbackAbort(Exception):
    pass


class _Status:
    def __init__(self, output_underflow: bool = False) -> None:
        self.input_underflow = False
        self.input_overflow = False
        self.output_underflow = output_underflow
        self.output_overflow = False
        self.priming_output = False


class _FakeWasapiSettings:
    def __init__(self, exclusive: bool = False) -> None:
        self.exclusive = exclusive


class _FakeOutputStream:
    def __init__(self, owner, **kwargs) -> None:
        self.owner = owner
        self.callback = kwargs["callback"]
        self.finished_callback = kwargs["finished_callback"]
        self.samplerate = float(kwargs["samplerate"])
        self.blocksize = int(kwargs["blocksize"])
        self.latency = 0.012

    def __enter__(self):
        block_index = 0
        callbacks_after_stop = self.owner.callbacks_after_stop
        while True:
            output = np.empty((self.blocksize, 1), dtype=np.float32)
            try:
                self.callback(
                    output,
                    self.blocksize,
                    {},
                    _Status(output_underflow=block_index == 1),
                )
            except _CallbackStop:
                self.owner.output_blocks.append(output.copy())
                if callbacks_after_stop > 0:
                    callbacks_after_stop -= 1
                    block_index += 1
                    continue
                self.finished_callback()
                break
            except _CallbackAbort:
                self.finished_callback()
                break
            self.owner.output_blocks.append(output.copy())
            block_index += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False


class _FakeSoundDevice:
    __version__ = "test"
    CallbackStop = _CallbackStop
    CallbackAbort = _CallbackAbort
    WasapiSettings = _FakeWasapiSettings

    def __init__(self, callbacks_after_stop: int = 0) -> None:
        self.output_blocks = []
        self.checked_settings = None
        self.callbacks_after_stop = callbacks_after_stop
        self.stream_arguments = None

    def query_devices(self, device=None, kind=None):
        devices = [
            {
                "name": "Input only",
                "hostapi": 0,
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
                "default_low_output_latency": 0.0,
                "default_high_output_latency": 0.0,
            },
            {
                "name": "Test output",
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
                "default_low_output_latency": 0.01,
                "default_high_output_latency": 0.1,
            },
        ]
        if device is None and kind is None:
            return devices
        return devices[1 if device is None else int(device)]

    def query_hostapis(self, index=None):
        hostapis = [{"name": "Windows WASAPI"}]
        return hostapis if index is None else hostapis[index]

    def check_output_settings(self, **kwargs) -> None:
        self.checked_settings = kwargs

    def OutputStream(self, **kwargs):
        self.stream_arguments = kwargs
        return _FakeOutputStream(self, **kwargs)

    def get_portaudio_version(self):
        return 190700, "Fake PortAudio"


class PortAudioBackendTests(unittest.TestCase):
    def test_lists_only_output_devices(self) -> None:
        devices = portaudio.list_output_devices(_FakeSoundDevice())
        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0]["index"], 1)
        self.assertEqual(devices[0]["name"], "Test output")
        self.assertEqual(devices[0]["hostapi"], "Windows WASAPI")

    def test_wasapi_exclusive_is_forwarded_to_output_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            fs_hz = 48000
            wav_path = tmp_path / "input.wav"
            samples = np.arange(300)
            _write_wav(
                wav_path,
                fs_hz,
                0.2 * np.sin(2.0 * np.pi * 100.0 * samples / fs_hz),
            )
            config = audio_engine.AudioRealtimeConfig(
                fs_hz=fs_hz,
                block_size=256,
            )
            fake_sd = _FakeSoundDevice()

            result = portaudio.run_portaudio_wav(
                wav_path,
                config,
                device=1,
                wasapi_exclusive=True,
                sounddevice_module=fake_sd,
            )

            settings = fake_sd.stream_arguments["extra_settings"]
            self.assertIsInstance(settings, _FakeWasapiSettings)
            self.assertTrue(settings.exclusive)
            self.assertTrue(result.backend.wasapi_exclusive)
            self.assertTrue(result.backend.summary()["wasapi_exclusive"])

    def test_portaudio_backend_matches_hardware_free_callback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            fs_hz = 48000
            samples = np.arange(700)
            source = 0.4 * np.sin(2.0 * np.pi * 173.0 * samples / fs_hz)
            wav_path = tmp_path / "input.wav"
            _write_wav(wav_path, fs_hz, source)
            config = audio_engine.AudioRealtimeConfig(
                fs_hz=fs_hz,
                block_size=256,
                preset_name="Spk-2",
                target_rms_voltage=1.0,
                output_mode="silent",
            )

            expected = audio_engine.run_simulated_wav(wav_path, config)
            fake_sd = _FakeSoundDevice()
            actual = portaudio.run_portaudio_wav(
                wav_path,
                config,
                device=1,
                sounddevice_module=fake_sd,
            )

            np.testing.assert_allclose(
                actual.audio.output_pressure_pa,
                expected.output_pressure_pa,
                rtol=0.0,
                atol=0.0,
            )
            self.assertEqual(actual.audio.statistics.block_count, 3)
            self.assertEqual(actual.backend.summary()["adapter_deadline_misses"], 0)
            self.assertEqual(
                actual.audio.statistics.summary()["driver_status"]["output_underflows"],
                1,
            )
            self.assertEqual(fake_sd.checked_settings["samplerate"], fs_hz)
            self.assertTrue(
                all(np.count_nonzero(block) == 0 for block in fake_sd.output_blocks)
            )

    def test_additional_driver_callback_after_completion_stops_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            fs_hz = 48000
            samples = np.arange(300)
            wav_path = tmp_path / "input.wav"
            _write_wav(
                wav_path,
                fs_hz,
                0.2 * np.sin(2.0 * np.pi * 100.0 * samples / fs_hz),
            )
            config = audio_engine.AudioRealtimeConfig(
                fs_hz=fs_hz,
                block_size=256,
            )
            fake_sd = _FakeSoundDevice(callbacks_after_stop=1)

            result = portaudio.run_portaudio_wav(
                wav_path,
                config,
                device=1,
                sounddevice_module=fake_sd,
            )

            self.assertEqual(result.audio.statistics.block_count, 2)
            self.assertEqual(len(fake_sd.output_blocks), 3)
            self.assertEqual(np.count_nonzero(fake_sd.output_blocks[-1]), 0)

    def test_pressure_monitor_reaches_driver_buffer_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            fs_hz = 48000
            samples = np.arange(700)
            wav_path = tmp_path / "input.wav"
            _write_wav(
                wav_path,
                fs_hz,
                0.3 * np.sin(2.0 * np.pi * 200.0 * samples / fs_hz),
            )
            config = audio_engine.AudioRealtimeConfig(
                fs_hz=fs_hz,
                block_size=256,
                output_mode="pressure",
                pressure_pa_per_full_scale=20.0,
                output_peak_limit=0.1,
            )
            fake_sd = _FakeSoundDevice()

            result = portaudio.run_portaudio_wav(
                wav_path,
                config,
                device=1,
                sounddevice_module=fake_sd,
            )

            driver_audio = np.concatenate(fake_sd.output_blocks, axis=0)[:, 0]
            np.testing.assert_array_equal(
                driver_audio[: result.audio.statistics.sample_count],
                result.audio.rendered_audio,
            )
            self.assertTrue(
                np.all(driver_audio[result.audio.statistics.sample_count :] == 0.0)
            )
            output_directory = tmp_path / "pressure_result"
            portaudio.write_portaudio_artifacts(
                result,
                output_directory,
                sounddevice_module=fake_sd,
            )
            metadata = json.loads(
                (output_directory / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                metadata["audio_output"]["scaled_pressure_monitor_is_audio_output"]
            )
            self.assertEqual(
                metadata["audio_output"]["monitor_diagnostics"]["mode"],
                "pressure",
            )

    def test_artifact_export_contains_pressure_timing_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            fs_hz = 44100
            samples = np.arange(300)
            wav_path = tmp_path / "input.wav"
            _write_wav(
                wav_path,
                fs_hz,
                0.3 * np.sin(2.0 * np.pi * 200.0 * samples / fs_hz),
            )
            config = audio_engine.AudioRealtimeConfig(fs_hz=fs_hz, block_size=128)
            fake_sd = _FakeSoundDevice()
            result = portaudio.run_portaudio_wav(
                wav_path,
                config,
                device=1,
                sounddevice_module=fake_sd,
            )

            output_directory = tmp_path / "result"
            portaudio.write_portaudio_artifacts(
                result,
                output_directory,
                sounddevice_module=fake_sd,
            )

            saved_pressure = np.load(output_directory / "output_pressure_pa.npy")
            np.testing.assert_array_equal(saved_pressure, result.audio.output_pressure_pa)
            timing_lines = (output_directory / "callback_timing.csv").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(
                len(timing_lines),
                result.audio.statistics.block_count + 1,
            )
            metadata = json.loads(
                (output_directory / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["audio_output"]["mode"], "silent")
            self.assertFalse(metadata["audio_output"]["pressure_is_audio_output"])
            self.assertEqual(
                metadata["portaudio_statistics"]["device"]["name"],
                "Test output",
            )
            self.assertEqual(metadata["software"]["sounddevice"], "test")


if __name__ == "__main__":
    unittest.main()
