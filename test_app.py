import unittest
from app import app
import os

class TestOCRApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.test_image = "test_image.png"
        with open(self.test_image, "wb") as f:
            f.write(b"fake image data")  # Replace with actual image for real testing

    def tearDown(self):
        if os.path.exists(self.test_image):
            os.remove(self.test_image)

    def test_index(self):
        response = self.app.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"TEXT EXTRACTION APP", response.data)

    def test_upload_image_invalid(self):
        response = self.app.post("/upload_image")
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"No files uploaded", response.data)

    def test_export_txt_no_text(self):
        response = self.app.get("/export_txt")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content_type, "text/plain")

if __name__ == "__main__":
    unittest.main()