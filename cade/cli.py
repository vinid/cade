# -*- coding: utf-8 -*-

"""Console script for cade."""
import sys
import click


@click.command()
def main(args=None):
    """Console script for cade."""
    click.echo("Replace this message by putting your code into "
               "cade.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
