def layout_full_zero_check(input_layout):
    row_check = input_layout.sum(dim=2).eq(0)  # (H, L // block)
    col_check = input_layout.sum(dim=1).eq(0)
    row_answer = bool(row_check.any())
    col_answer = bool(col_check.any())
    return row_answer or col_answer, row_answer, col_answer, row_check, col_check
