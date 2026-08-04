"""Unit tests for fluxgen.image_validation.validate_image_file."""
from pathlib import Path

import pytest
from PIL import Image as PILImage
from PIL import Image as PILImage_mod  # alias for monkeypatch.setattr below

from fluxgen.exceptions import InvalidImageError
from fluxgen.image_validation import validate_image_file


# ── Resolved path ────────────────────────────────────────────────────────────


def test_validate_image_file_resolves_path(tmp_path):
    """Returned path is absolute + expanded + resolved."""
    image = tmp_path / "a.png"
    PILImage.new("RGB", (10, 10), "red").save(image)

    resolved = validate_image_file(str(image))

    assert isinstance(resolved, Path)
    assert resolved.is_absolute()
    assert resolved == image.resolve()


def test_validate_image_file_expands_user(monkeypatch, tmp_path):
    """`~` is expanded before resolution."""
    # Use HOME monkeypatch to point at tmp_path so `~` becomes tmp_path.
    monkeypatch.setenv("HOME", str(tmp_path))

    home_image = tmp_path / "from_home.png"
    PILImage.new("RGB", (10, 10), "blue").save(home_image)

    resolved = validate_image_file("~/from_home.png")

    assert resolved == home_image.resolve()


def test_validate_image_file_accepts_pathlike(tmp_path):
    """`os.PathLike` (pathlib.Path) is accepted, not only `str`."""
    image = tmp_path / "a.png"
    PILImage.new("RGB", (10, 10), "red").save(image)

    resolved = validate_image_file(image)  # pass Path, no str()

    assert resolved == image.resolve()


# ── read_size contract ──────────────────────────────────────────────────────


def test_validate_image_file_with_read_size_returns_dimensions(tmp_path):
    image = tmp_path / "a.png"
    PILImage.new("RGB", (640, 480), "red").save(image)

    resolved, size = validate_image_file(str(image), read_size=True)

    assert resolved == image.resolve()
    assert size == (640, 480)


def test_validate_image_file_opens_exactly_once(tmp_path, monkeypatch):
    """size + verify happen in the same PIL.Image.open() — one open total."""
    image = tmp_path / "a.png"
    PILImage.new("RGB", (100, 200), "red").save(image)

    real_open = PILImage_mod.open
    open_count = 0

    def counting_open(*args, **kwargs):
        nonlocal open_count
        open_count += 1
        return real_open(*args, **kwargs)

    monkeypatch.setattr(PILImage_mod, "open", counting_open)

    resolved, size = validate_image_file(str(image), read_size=True)

    assert size == (100, 200)
    assert open_count == 1


def test_validate_image_file_does_not_open_twice_when_read_size_false(
    tmp_path, monkeypatch
):
    """read_size=False opens once for verify() — not twice."""
    image = tmp_path / "a.png"
    PILImage.new("RGB", (50, 50), "green").save(image)

    real_open = PILImage_mod.open
    open_count = 0

    def counting_open(*args, **kwargs):
        nonlocal open_count
        open_count += 1
        return real_open(*args, **kwargs)

    monkeypatch.setattr(PILImage_mod, "open", counting_open)

    resolved = validate_image_file(str(image))

    assert open_count == 1
    assert resolved == image.resolve()


# ── Custom label contract ──────────────────────────────────────────────────


def test_validate_image_file_label_reprefixes_filenotfound(tmp_path):
    """`label=` restores caller-specific "X not found: ..." wording."""
    missing = tmp_path / "nope.png"

    with pytest.raises(FileNotFoundError, match=r"Reference image not found"):
        validate_image_file(str(missing), label="reference image")


def test_validate_image_file_label_reprefixes_value_error(tmp_path):
    """`label=` also affects the directory-not-a-file ValueError."""
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    with pytest.raises(ValueError, match=r"Reference image must be a file"):
        validate_image_file(str(subdir), label="reference image")


def test_validate_image_file_default_label_is_image(tmp_path, monkeypatch):
    """Without `label=`, default messages use 'Image' / 'image'."""
    missing = tmp_path / "nope.png"
    with pytest.raises(FileNotFoundError, match=r"Image not found"):
        validate_image_file(str(missing))


# ── Locked error message contracts ──────────────────────────────────────────


def test_validate_image_file_filenotfound_error_message_contract(tmp_path):
    """Lock the default-`label` FileNotFoundError message format."""
    missing = tmp_path / "no.png"
    with pytest.raises(FileNotFoundError) as excinfo:
        validate_image_file(str(missing))

    msg = str(excinfo.value)
    assert msg.startswith("Image not found: ")
    assert str(missing.resolve()) in msg


def test_validate_image_file_value_error_message_contract(tmp_path):
    """Lock the ValueError message when path is a directory."""
    subdir = tmp_path / "dir"
    subdir.mkdir()
    with pytest.raises(ValueError) as excinfo:
        validate_image_file(str(subdir))

    msg = str(excinfo.value)
    assert msg.startswith("Image must be a file: ")
    assert str(subdir.resolve()) in msg


def test_validate_image_file_unidentified_image_error_contract(tmp_path):
    """Lock the InvalidImageError message for UnidentifiedImageError."""
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"this is not an image file")

    with pytest.raises(InvalidImageError) as excinfo:
        validate_image_file(str(corrupt))

    msg = str(excinfo.value)
    assert msg.startswith("Invalid or corrupted image file: ")
    assert str(corrupt.resolve()) in msg


def test_validate_image_file_oserror_during_verify_contract(tmp_path, monkeypatch):
    """Lock the InvalidImageError message for non-UnidentifiedImageError verify failures.

    Forces the (OSError, Image.DecompressionBombError) branch by
    patching Image.verify to raise OSError. Empty/zero-byte files all
    raise UnidentifiedImageError (caught first), so this branch needs
    a synthetic injection.
    """
    image = tmp_path / "a.png"
    PILImage.new("RGB", (10, 10), "red").save(image)

    real_open = PILImage_mod.open

    class _FlakyImage:
        """Proxies to a real PIL image but overrides .verify()."""

        def __init__(self, inner):
            self._inner = inner

        @property
        def size(self):
            return self._inner.size

        def verify(self):
            raise OSError("simulated decode failure")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def flaky_open(*args, **kwargs):
        return _FlakyImage(real_open(*args, **kwargs))

    monkeypatch.setattr(PILImage_mod, "open", flaky_open)

    with pytest.raises(InvalidImageError) as excinfo:
        validate_image_file(str(image))

    msg = str(excinfo.value)
    assert msg.startswith("Could not verify image file ")
    assert str(image.resolve()) in msg
    assert "simulated decode failure" in msg


# ── Backwards-compat smoke checks for the inputs touched by the refactor ──


def test_validate_image_file_raises_filenotfound_for_missing(tmp_path):
    missing = tmp_path / "nope.png"
    with pytest.raises(FileNotFoundError, match="Image not found"):
        validate_image_file(str(missing))


def test_validate_image_file_rejects_directory(tmp_path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    with pytest.raises(ValueError, match="must be a file"):
        validate_image_file(str(subdir))


def test_validate_image_file_rejects_corrupt_file(tmp_path):
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not a real image")

    with pytest.raises(InvalidImageError, match="Invalid or corrupted image"):
        validate_image_file(str(corrupt))


def test_validate_image_file_rejects_zero_byte_file(tmp_path):
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")

    with pytest.raises(InvalidImageError):
        validate_image_file(str(empty))
