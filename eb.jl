using Distributions

function Binary_EBP_minibatch(x_in, d_in, enable, K, h0_cell, bias0_cell, eta, sigma_w, dropout, batch_size)
    # EBP algorithm for multilayer neural network with binary weights using minibatch and dropout options.
    #
    # Inputs:
    #   x_in      : T×M input matrix, where T is the time length (number of patterns)
    #   d_in      : T×N label matrix
    #   enable    : training flag (0 or 1)
    #   K         : vector of length L+1; the l-th component is the width of the l-th layer, L is the number of layers
    #   h0_cell   : initial hidden weights (vector of matrices) or empty array
    #   bias0_cell: initial biases (vector of matrices) or empty array
    #   eta       : learning rate (set to 1 for testing purposes)
    #   sigma_w   : parameter proportional to the variance of the initial weights
    #   dropout   : flag (0 or 1) to apply dropout on input
    #   batch_size: mini-batch size
    #
    # Outputs:
    #   r_out_cont: T×N output of network with probabilistic (EBP-P) rule
    #   r_out     : T×N output of network with deterministic (EBP-D) rule
    #   h_cell    : final hidden weights (vector of matrices)
    #   bias_cell : final biases (vector of matrices)
    
    T = size(x_in, 1)
    if size(d_in, 1) != T
        error("The length of 'x' and 'd' should be equal!")
    end
    L = length(K) - 1

    # Transpose input and labels (from row to column vectors)
    x = x_in'
    y = d_in'

    # Initialize output arrays (same size as d_in)
    r_out = zeros(size(d_in))
    r_out_cont = zeros(size(d_in))

    # Initialize containers (using vectors to mimic MATLAB cell arrays)
    mean_v_prev_cell = Vector{Any}(undef, L+1)
    mean_u_cell       = Vector{Any}(undef, L)
    var_u_cell        = Vector{Any}(undef, L)

    # Initialize weights and tanh(weights)
    if isempty(h0_cell)
        h_cell = Vector{Any}(undef, L)
        tanh_h_cell = Vector{Any}(undef, L)
        for ll in 1:L
            # Note: rand(K[ll+1], K[ll]) gives a matrix with uniform random numbers in [0,1).
            h_cell[ll] = (rand(K[ll+1], K[ll]) .- 0.5) .* sqrt(sigma_w * 12 / K[ll])
            tanh_h_cell[ll] = tanh.(h_cell[ll])
        end
    else
        h_cell = h0_cell
        tanh_h_cell = Vector{Any}(undef, L)
        for ll in 1:L
            tanh_h_cell[ll] = tanh.(h_cell[ll])
        end
    end

    # Initialize biases
    if isempty(bias0_cell)
        bias_cell = Vector{Any}(undef, L)
        for ll in 1:L
            bias_cell[ll] = zeros(K[ll+1], 1)
        end
    else
        bias_cell = bias0_cell
    end

    batch_size_fixed = batch_size
    time_progress = 0.0

    # Process mini-batches
    for tt in 1:batch_size_fixed:T
        current_batch_size = min(batch_size, T - tt + 1)
        batch_ind = tt:(tt + current_batch_size - 1)

        #### Forward pass
        # First layer:
        mean_v = x[:, batch_ind]
        tanh_h = tanh_h_cell[1]
        bias = bias_cell[1]
        mean_u = (tanh_h * mean_v .+ bias) / sqrt(K[1] + 1)

        if dropout != 0
            p_in = 0.8  # dropout probability
            var_input = 4 * p_in * (1 - p_in)  # variance for binary ±1 Bernoulli noise
            var_u = ((1 .- (1 - var_input) .* (tanh_h.^2)) * (mean_v.^2) .+ 1) / (K[1] + 1)
        else
            var_u = ((1 .- tanh_h.^2) * (mean_v.^2) .+ 1) / (K[1] + 1)
        end

        p_v = cdf.(Normal(0, 1), mean_u ./ sqrt.(var_u))
        mean_u_cell[1] = mean_u
        var_u_cell[1] = var_u
        mean_v_prev_cell[1] = mean_v

        mean_v = 2 .* p_v .- 1
        var_v = 4 .* (p_v .- p_v.^2)
        mean_v_prev_cell[2] = mean_v

        # Layers 2 through L:
        for ll in 2:L
            bias = bias_cell[ll]
            tanh_h = tanh_h_cell[ll]
            mean_u = (tanh_h * mean_v .+ bias) / sqrt(K[ll] + 1)
            # In MATLAB, bsxfun adds a row vector to each row of the matrix.
            # Here sum(var_v, dims=1) produces a 1×batch_size row vector, which broadcasts correctly.
            var_u = (sum(var_v, dims=1) .+ ((1 .- tanh_h.^2) * (1 .- var_v) .+ 1)) / (K[ll] + 1)
            p_v = cdf.(Normal(0, 1), mean_u ./ sqrt.(var_u))
            mean_u_cell[ll] = mean_u
            var_u_cell[ll] = var_u
            mean_v = 2 .* p_v .- 1
            var_v = 4 .* (p_v .- p_v.^2)
            mean_v_prev_cell[ll+1] = mean_v
        end

        # Probabilistic output (EBP-P)
        r_out_cont[batch_ind, :] .= mean_v'

        # Deterministic output (EBP-D)
        v = x[:, batch_ind]
        for ll in 1:(L-1)
            h = h_cell[ll]
            bias = bias_cell[ll]
            v = sign.(h) * v
            v = sign.(v .+ bias)
        end
        h_last = h_cell[L]
        bias_last = bias_cell[L]
        v = (sign.(h_last) * v) .+ bias_last
        r_out[batch_ind, :] .= v'

        #### Backward pass
        if enable == 1
            for ll in L:-1:1
                mean_v_prev = mean_v_prev_cell[ll]
                mean_u = mean_u_cell[ll]
                var_u = var_u_cell[ll]
                bias = bias_cell[ll]
                h = h_cell[ll]
                tanh_h = tanh_h_cell[ll]

                if ll == L
                    Y = y[:, batch_ind]
                    # Compute Gi using element-wise normal PDF and CDF evaluations.
                    Gi = 2 .* (pdf.(Normal.(mean_u, sqrt.(var_u)), 0) ./ cdf.(Normal.(-Y .* mean_u, sqrt.(var_u)), 0)) / sqrt(K[ll] + 1)
                    # Replace non-finite Gi values.
                    Gi = ifelse.(isfinite.(Gi), Gi,
                        -2 .* ((Y .* mean_u .< 0) .* (mean_u ./ var_u)) ./ sqrt(K[ll] + 1))
                    delta_next = Y
                else
                    delta_next = delta  # delta from the previous (deeper) layer
                    Gi = 2 .* pdf.(Normal.(mean_u, sqrt.(var_u)), 0) / sqrt(K[ll] + 1)
                end

                delta = transpose(tanh_h) * (delta_next .* Gi)
                h .= h .+ 0.5 * eta * (delta_next .* Gi) * transpose(mean_v_prev)
                h_cell[ll] = h
                tanh_h_cell[ll] = tanh.(h)
                bias .= bias .+ 0.5 * eta * sum(delta_next .* Gi, dims=2)
                bias_cell[ll] = bias
            end
        elseif enable == 0
            # No weight update when enable==0.
        else
            error("enable input must be 0 or 1")
        end

        # Progress display (every ~1% complete)
        ratio = 0.01
        temp = ratio * floor((tt/T) / ratio)
        if temp > time_progress
            time_progress = temp
            println("$(100 * temp)% complete")
        end

    end

    return r_out_cont, r_out, h_cell, bias_cell
end
