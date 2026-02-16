import unittest

from src.openly.pii_guard import redact_payload, redact_text


class TestPiiGuard(unittest.TestCase):
    def test_redact_text_masks_email_and_phone(self):
        txt = "Contact me at mom@test.com or +91 99999 11111"
        redacted = redact_text(txt)
        self.assertNotIn("mom@test.com", redacted)
        self.assertIn("[REDACTED_EMAIL]", redacted)
        self.assertIn("[REDACTED_PHONE]", redacted)

    def test_redact_payload_masks_sensitive_keys(self):
        payload = {
            "parent_name": "Asha",
            "notes": "email asha@x.com",
            "age": 5,
        }
        redacted = redact_payload(payload)
        self.assertEqual(redacted["parent_name"], "[REDACTED]")
        self.assertNotIn("asha@x.com", redacted["notes"])
        self.assertEqual(redacted["age"], 5)


if __name__ == "__main__":
    unittest.main()
