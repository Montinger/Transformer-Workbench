"""Defines some general utils (non model specific) for the LoRA training
"""

import math
import torch
from torch import nn
from torch import Tensor

import tabulate
import tokenizers

def count_parameters(model):
    """
    Counts the number of trainable parameters of a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model for which the number of trainable parameters will be counted.

    Returns
    -------
    int
        The number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def format_with_underscore(n):
    """Mini helper function to format a number with underscore as thousand separator"""
    return f"{n:_}"


def parameter_count_table(model, output_file_path=None, output_print=True, add_dtypes=False, show_nograd_paras=False):
    """
    Displays a formatted table containing the number of trainable parameters
    for each named element of a PyTorch model, along with the total count.

    This function uses the `tabulate` library to create a table that lists
    the number of trainable parameters in each module of the provided PyTorch
    model. The table is formatted with two columns: 'Module' and 'Parameters'.
    It displays the module's name in the 'Module' column and the number of
    trainable parameters in the 'Parameters' column.

    The function also calculates the total number of trainable parameters
    in the model and appends this information to the table.

    The table is printed to the console if `output_print` is True. If `output_file_path`
    is provided, the table is also written to that file.

    The numbers are printed with an underscore as a thousand separator.

    Parameters
    ----------
    model : torch.nn.Module
        The PyTorch model for which the parameter count table should be generated.
    output_file_path : str or Path, optional
        The path to the file where the table should be written. If not provided, the table
        is only printed to the console.
    output_print : bool, optional
        If True, the table is printed to the console. Default is True.
    add_dtypes : bool, optional
        If True also outputs the dtypes for each parameter set. Default is False.
    show_nograd_paras : bool, optional
        If True also shows parameters which are frozen, i.e. have no gradient. Default is False.

    Returns
    -------
    None
        The function does not return any value; it prints or writes the table.

    Requires
    --------
    tabulate : Python package
        The `tabulate` package is required to format and print the table.

    Examples
    --------
    >>> import torch
    >>> import torch.nn as nn
    >>> model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))
    >>> parameter_count_table(model)
    """
    table = [["Module", "Parameters"]]
    if add_dtypes:
        table = [["Module", "Parameters", "dtype"]]
    total_params = 0
    max_len = 0
    for name, parameter in model.named_parameters():
        if (not parameter.requires_grad) and (not show_nograd_paras): continue
        params = parameter.numel()
        formatted_params = format_with_underscore(params)
        max_len = max(max_len, len(formatted_params))
        if add_dtypes:
            table.append([str(name), formatted_params, parameter.dtype])
        else:
            table.append([str(name), formatted_params])
        total_params += params

    table.append(tabulate.SEPARATING_LINE)

    formatted_total = format_with_underscore(total_params)
    max_len = max(max_len, len(formatted_total))
    if add_dtypes:
        table.append(["TOTAL", formatted_total])
    else:
        table.append(["TOTAL", formatted_total, ''])

    # Right align the numbers in the table
    for row in table[1:]:
        if row is not tabulate.SEPARATING_LINE:
            row[1] = row[1].rjust(max_len)

    tabulated_table = tabulate.tabulate(table, headers="firstrow")
    if output_file_path is not None:
        with open(output_file_path, 'w') as f:
            f.write(tabulated_table)
    if output_print:
        print(tabulated_table)
        print("")



def get_lr(optimizer):
    """
    Get the learning rate of an optimizer.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The PyTorch optimizer instance whose learning rate needs to be retrieved.

    Returns
    -------
    float
        The learning rate of the optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
