"""Minimal Fubon login smoke test using environment variables.

For realtime market data collection, use:
    python -m realtime.main
"""

from realtime.config import RealtimeConfig


def main() -> None:
    from fubon_neo.sdk import FubonSDK

    config = RealtimeConfig.from_env()
    config.validate()

    sdk = FubonSDK()
    accounts = sdk.login(
        config.fubon_id,
        config.fubon_password,
        config.fubon_cert_path,
        config.fubon_cert_password_for_login,
    )
    print(accounts)


if __name__ == "__main__":
    main()
