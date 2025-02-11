from pywxdump.db import MediaHandler

config1 = {
    "key": "test1",
    "type": "sqlite",
    "path": r"D:\projects\python projects\WeClone-data\merge_1737893929.db",
}


t1 = MediaHandler(config1)
t1.get_audio(
    "9219950799821647157",
    is_play=True,
    is_wave=True,
    save_path=r"D:\projects\python projects\WeClone-data\test.wav",
)


