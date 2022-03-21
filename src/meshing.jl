spacing(::Periodic, n) = 1 / n
spacing(::NonPeriodic, n) = 1 / (n-1)

coordinate(top, n) = UnitRange(0, n-1) .* spacing(top, n)

