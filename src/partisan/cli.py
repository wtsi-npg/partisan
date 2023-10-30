# -*- coding: utf-8 -*-
#
# Copyright © 2022, 2023 Genome Research Ltd. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# @author Keith James <kdj@sanger.ac.uk>

import logging
from pathlib import PurePath
from pprint import PrettyPrinter

import click
from click import ParamType
from click.shell_completion import CompletionItem
from structlog import get_logger

import partisan
from partisan.irods import Collection, DataObject, make_rods_item, rods_path_type

logging.basicConfig(level=logging.ERROR)
log = get_logger()


class RodsPathType(ParamType):
    """A Click parameter type representing an iRODS path."""

    name = "rods_path"

    def shell_complete(self, ctx, param, incomplete):
        path = PurePath(incomplete)
        if not path.is_absolute():
            return [CompletionItem("/")]

        match rods_path_type(path):
            case partisan.irods.DataObject:
                return [CompletionItem(path)]
            case partisan.irods.Collection:
                return [
                    CompletionItem(s)
                    for p in Collection(path).contents()
                    if (s := str(p)).startswith(incomplete)
                ]
            case _:
                parent = path.parent

                match rods_path_type(parent):
                    case partisan.irods.Collection:
                        return [
                            CompletionItem(s)
                            for p in Collection(parent).contents()
                            if (s := str(p)).startswith(incomplete)
                        ]
                    case _:
                        return [CompletionItem("/")]


class LsPrinter(PrettyPrinter):
    """A proof-of-concept iRODS path pretty printer."""

    @staticmethod
    def _format_obj(obj):
        sep = " "
        size = obj.size()
        timestamp = obj.timestamp()
        checksum = obj.checksum()
        replicas = obj.replicas()
        return sep.join(
            [
                f"{checksum} {len(replicas):>2}",
                f"{size:>12}",
                f"{timestamp}",
                f"{obj}",
            ],
        )

    @staticmethod
    def _format_col(col):
        pad = " " * 75
        return pad + str(col)

    def format(self, obj, context, maxlevels, level):
        if isinstance(obj, DataObject):
            return self._format_obj(obj), False, False
        elif isinstance(obj, Collection):
            context[id(obj)] = 1
            recur = False

            contents = obj.contents()
            if contents:
                recur = True

            sep = "\n"
            parts = [self._format_col(obj)]
            for elt in contents:
                s, _, _ = LsPrinter.format(self, elt, context, maxlevels, level - 1)
                parts.append(s)

            return sep.join(parts), False, recur
        else:
            return PrettyPrinter.format(self, obj, context, maxlevels, level)


@click.command()
@click.argument("rods_path", type=RodsPathType())
@click.option("-l", is_flag=True, help="Use a long listing format")
def pls(rods_path, l=False):
    item = make_rods_item(rods_path)
    if l:
        click.echo(LsPrinter(depth=1, indent=10).pprint(item))
    else:
        click.echo(str(item))
