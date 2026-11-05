const ϕ⁻¹ = (√(5.) - 1.) / 2.
const ϕ⁻² = (3. - √(5.)) / 2.

function gss(f, a, b; tol = 1e-5)
    Δ = b - a
    
    n = ceil(Int, log(tol / Δ) / log(ϕ⁻¹))
    
    c = a + Δ * ϕ⁻²
    d = a + Δ * ϕ⁻¹

    yc = f(c)
    yd = f(d)

    for _ in 1:n
        if yc < yd
            b = d
            d, yd = c, yc
            Δ *= ϕ⁻¹
            c = a + ϕ⁻² * Δ
            yc = f(c)
        else
            a = c
            c, yc = d, yd
            Δ *= ϕ⁻¹
            d = a + Δ * ϕ⁻¹
            yd = f(d)
        end
    end

    y, x = yc < yd ? (yc, (a + d) / 2) :  (yd, (c + b) / 2)
    
    y₀ = f(a)
    y₁ = f(b)

    values = (y₀, y, y₁)
    candidates = (a, x, b)

    optidx = argmax(values)

    return values[optidx], candidates[optidx]
end