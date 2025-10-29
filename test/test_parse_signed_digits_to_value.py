import os, sys, pathlib, pytest

# 确保能 import 到项目里的 auto.drivers.mac
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auto.drivers.mac import _parse_signed_digits_to_value

@pytest.mark.parametrize("s,expected", [
    ("+2811", 28.11),
    ("+092", 0.92),
    ("-1024", -10.24),
    ("+000", 0.00),
    ("-789", -7.89),
    ("+1234", 12.34),
    ("-005", -0.05),
    ("+010", 0.10),
    ("-999", -9.99),
    ("+450", 4.50),
    ("+1000", 10.00),
    ("-1203", -12.03),
    (" + 2 0 3 ", 2.03),  # 去空格后有效
])
def test_ok(s, expected):
    assert _parse_signed_digits_to_value(s) == pytest.approx(expected, abs=1e-6)

@pytest.mark.parametrize("s", [
    "-+2901",   # 符号非法
    "-02",      # 位数不足
    "123",      # 无符号
    "+12345",   # 位数过多
    "-4-",      # 乱序
    "++203",    # 符号重复
    None,       # 空
])
def test_invalid_format(s):
    with pytest.raises(RuntimeError):
        _parse_signed_digits_to_value(s)  # 应抛格式错误

@pytest.mark.parametrize("s", [
    "+6500",  # 65.00 > 64
    "-6401",  # -64.01 < -64
])
def test_out_of_range(s):
    with pytest.raises(RuntimeError):
        _parse_signed_digits_to_value(s)  # 越界应抛错