# Price change different

Use `dP = ln(P[t+1]) - ln(P[t])`

Also `= ln(P[t+1] / P[t])`

    a = np.diff(np.log(y))

Anti:

    y = exp(ln(y0) + a) = y0 * exp(a)
