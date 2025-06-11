import argparse
import os

from pywxdump.db import MediaHandler


def main():
    parser = argparse.ArgumentParser(description="Extract audio from WeChat database")
    parser.add_argument("--db-path", type=str, required=True, help="Path to WeChat database file")
    parser.add_argument("--MsgSvrID", type=str, required=True, help="Message server ID of the audio")
    parser.add_argument(
        "--save-path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "sample.wav"),
        help="Path to save the audio file (default: sample.wav in script directory)",
    )
    parser.add_argument(
        "--rate", type=int, default=24000, help="Sample rate for audio conversion (default: 24000)"
    )

    args = parser.parse_args()

    config = {
        "key": "test1",
        "type": "sqlite",
        "path": args.db_path,
    }

    t1 = MediaHandler(config)
    t1.get_audio(
        MsgSvrID=args.MsgSvrID,
        is_play=True,
        is_wave=True,
        save_path=args.save_path,
        rate=args.rate,
    )


if __name__ == "__main__":
    main()
