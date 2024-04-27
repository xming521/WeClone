def dict_to_argv(d):
    argv = []
    for k, v in d.items():
        argv.append('--' + k)
        if v is not None:
            argv.append(str(v))
    return argv


