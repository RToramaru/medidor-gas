import unittest

from leitor_gas.cameras import find_cameras


class CaptureStub:
    def __init__(self, index, available):
        self.index = index
        self.available = available
        self.released = False

    def isOpened(self):
        return self.available

    def release(self):
        self.released = True


class CameraTests(unittest.TestCase):
    def test_finds_available_cameras_and_releases_every_probe(self):
        captures = []

        def capture_factory(index):
            capture = CaptureStub(index, index in {1, 3})
            captures.append(capture)
            return capture

        available = find_cameras(max_devices=5, capture_factory=capture_factory)

        self.assertEqual(available, [1, 3])
        self.assertTrue(all(capture.released for capture in captures))


if __name__ == "__main__":
    unittest.main()
