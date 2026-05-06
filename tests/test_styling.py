import unittest
from fluxgen.styling import StyleManager

class TestStyling(unittest.TestCase):
    def test_none_style_returns_raw_prompt(self):
        sm = StyleManager()
        self.assertEqual(sm.apply_style("A cat", "none"), "A cat")

    def test_empty_style_returns_raw_prompt(self):
        sm = StyleManager()
        self.assertEqual(sm.apply_style("A cat", ""), "A cat")

    def test_ghibli_style(self):
        sm = StyleManager()
        self.assertEqual(
            sm.apply_style("A cat", "ghibli"),
            "A cat in Studio Ghibli style, whimsical animation"
        )

    def test_cinematic_style(self):
        sm = StyleManager()
        self.assertEqual(
            sm.apply_style("A cat", "cinematic"),
            "A cat cinematic lighting, 8k resolution, highly detailed"
        )

    def test_all_builtin_styles_exist(self):
        sm = StyleManager()
        expected = [
            "none", "ghibli", "cinematic", "pixel", "watercolor",
            "anime", "photorealistic", "oil-painting", "comic",
            "minimal", "cyberpunk",
        ]
        for style in expected:
            result = sm.apply_style("test", style)
            self.assertIsInstance(result, str)

    def test_builtin_styles_modify_prompt(self):
        sm = StyleManager()
        for style in ["ghibli", "cinematic", "pixel", "watercolor",
                       "anime", "photorealistic", "oil-painting",
                       "comic", "minimal", "cyberpunk"]:
            result = sm.apply_style("test", style)
            self.assertNotEqual(result, "test", f"Style '{style}' should modify the prompt")

    def test_custom_styles(self):
        sm = StyleManager({"retro": " in retro 80s style"})
        self.assertEqual(
            sm.apply_style("A cat", "retro"),
            "A cat in retro 80s style"
        )

    def test_custom_style_overrides_builtin(self):
        sm = StyleManager({"ghibli": " custom ghibli override"})
        self.assertEqual(
            sm.apply_style("A cat", "ghibli"),
            "A cat custom ghibli override"
        )

    def test_unknown_style_returns_raw_prompt(self):
        sm = StyleManager()
        self.assertEqual(sm.apply_style("A cat", "unknown"), "A cat")

    def test_case_insensitive(self):
        sm = StyleManager()
        self.assertEqual(
            sm.apply_style("A cat", "Ghibli"),
            sm.apply_style("A cat", "ghibli"),
        )

    def test_get_style_names(self):
        sm = StyleManager()
        names = sm.get_style_names()
        self.assertIn("ghibli", names)
        self.assertIn("none", names)
        self.assertIn("cyberpunk", names)
        self.assertEqual(len(names), 11)

if __name__ == "__main__":
    unittest.main()
