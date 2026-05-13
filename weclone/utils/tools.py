def dict_to_argv(d):
    argv = []
    for k, v in d.items():
        if v is None:
            continue
        argv.append("--" + k)
        argv.append(str(v))
    return argv
