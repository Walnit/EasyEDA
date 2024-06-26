def parse_schedule(filename):
    parsed = []
    with open(filename) as f:
        lines = f.read().splitlines()
        parsed = [ line.split() for line in lines ]
    return parsed


