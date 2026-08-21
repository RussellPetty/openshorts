import unittest

from url_utils import build_video_url, video_filename_from_url


class BuildVideoUrlTests(unittest.TestCase):
    def test_encodes_reserved_and_unicode_filename_characters(self):
        url = build_video_url(
            "job-id",
            "The Whole Perspective Live – Episode #2_clip_1.mp4",
        )

        self.assertEqual(
            url,
            "/videos/job-id/The%20Whole%20Perspective%20Live%20%E2%80%93%20Episode%20%232_clip_1.mp4",
        )

    def test_encodes_job_id_as_a_single_path_segment(self):
        self.assertEqual(
            build_video_url("job/with/slashes", "clip.mp4"),
            "/videos/job%2Fwith%2Fslashes/clip.mp4",
        )

    def test_recovers_encoded_filename_for_disk_access(self):
        self.assertEqual(
            video_filename_from_url('/videos/job/Episode%20%232_clip_1.mp4'),
            'Episode #2_clip_1.mp4',
        )

    def test_rejects_encoded_path_traversal(self):
        with self.assertRaises(ValueError):
            video_filename_from_url('/videos/job/..%2Fsecret.mp4')


if __name__ == "__main__":
    unittest.main()
