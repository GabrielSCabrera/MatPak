from subprocess import call

def call_backend(program = "main"):
    """
        Opens the desired C++ script located in the "backend" directory.
        It will also pass the name of the simulation datafile to be read by
        the C++ backend as a command line argument.
    """
    call(["./{}".format(program)])

call_backend()
