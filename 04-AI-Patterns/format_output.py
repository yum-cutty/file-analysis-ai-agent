def print_box(title: str, lines: list[str], width: int = 60) -> None:
    """
    Utility function to print a box around the result for better visibility in console.
    
    :param title: the string to be displayed as the title of the box
    :type title: str
    :param lines: the list of strings to be displayed inside the box
    :type lines: list[str]
    :param width: the width of the box
    :type width: int
    """
    
    border = "█" * width
    empty  = "█" + " " * (width - 2) + "█"

    print("\n")
    print(border)
    print(empty)
    print("█" + title.center(width - 2) + "█")
    print(empty)
    print("█" + ("─" * (width - 2)) + "█")
    print(empty)

    for line in lines:
        while len(line) > width - 4:
            chunk = line[:width - 4]
            print("█ " + chunk.ljust(width - 4) + " █")
            line = line[width - 4:]
        print("█ " + line.ljust(width - 4) + " █")

    print(empty)
    print(border)
    print("\n")