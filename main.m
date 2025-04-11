function video_compression_demo()
    % Video Compression Demo in MATLAB with Multithreading
    % Implements DCVC, Hybrid Entropy, SlimVC, INR, and ROI techniques
    % Date: April 11, 2025

    % Load video (assumes MP4 input, e.g., beauty.mp4)
    video_path = 'ShakeNDry.mp4'; % Update to your file path
    vid_obj = VideoReader(video_path);
    max_frames = 10;
    frames = cell(max_frames, 1);
    for i = 1:min(max_frames, vid_obj.NumFrames)
        frame = readFrame(vid_obj);
        frames{i} = rgb2gray(frame); % Convert to grayscale
    end

    if isempty(frames{1})
        disp('Error: No frames loaded.');
        return;
    end

    % Compression settings
    quant_levels = [5, 20]; % SlimVC-like
    key_frame_interval = 5; % INR-like
    compressed_data = cell(max_frames, 1);
    key_frames = containers.Map('KeyType', 'int32', 'ValueType', 'any');
    prev_frames = cell(max_frames, 1); % Store previous frames for parfor

    % Prepare previous frames (sequential due to dependency)
    for i = 1:length(frames)
        if i > 1
            prev_frames{i} = double(frames{i-1});
        else
            prev_frames{i} = [];
        end
    end

    % Compress frames in parallel
    parfor i = 1:length(frames)
        frame = double(frames{i});
        quant_step = quant_levels(mod(i-1, 2) + 1);
        prev_frame = prev_frames{i};
        
        if mod(i-1, 4) == 0 % DCVC
            compressed_data{i} = compress_frame_dcvc(frame, quant_step, prev_frame);
        elseif mod(i-1, 4) == 1 % Entropy
            compressed_data{i} = compress_frame_entropy(frame, quant_step, prev_frame);
        elseif mod(i-1, 4) == 2 % SlimVC
            compressed_data{i} = compress_frame_slimvc(frame, quant_step);
        else % ROI
            compressed_data{i} = compress_frame_roi(frame, quant_step, prev_frame);
        end
    end

    % Sequential key frame assignment
    for i = 1:length(frames)
        if mod(i-1, key_frame_interval) == 0
            key_frames(i) = frames{i};
        end
    end

    % Decompress frames in parallel
    recon_frames = cell(max_frames, 1);
    prev_recons = cell(max_frames, 1); % Store previous reconstructed frames
    for i = 1:length(frames)
        if i > 1
            prev_recons{i} = double(recon_frames{i-1});
        else
            prev_recons{i} = [];
        end
    end

    parfor i = 1:length(compressed_data)
        data = compressed_data{i};
        prev_recon = prev_recons{i};
        if mod(i-1, 4) == 0 % DCVC
            quant_frame = data{1}; frame_shape = data{2}; quant_step = data{3};
            recon_frames{i} = decompress_frame_dcvc(quant_frame, frame_shape, quant_step, prev_recon);
        elseif mod(i-1, 4) == 1 % Entropy
            encoded = data{1}; frame_shape = data{2}; quant_step = data{3}; codebook = data{4};
            recon_frames{i} = decompress_frame_entropy(encoded, frame_shape, quant_step, codebook, prev_recon);
        elseif mod(i-1, 4) == 2 % SlimVC
            quant_frame = data{1}; frame_shape = data{2}; quant_step = data{3};
            recon_frames{i} = decompress_frame_dcvc(quant_frame, frame_shape, quant_step, prev_recon);
        else % ROI
            quant_frame = data{1}; frame_shape = data{2}; quant_step = data{3}; roi_mask = data{4};
            recon_frames{i} = decompress_frame_roi(quant_frame, frame_shape, quant_step, roi_mask, prev_recon);
        end
        % INR override for non-key frames
        if ~isKey(key_frames, i)
            recon_frames{i} = decompress_frame_inr(key_frames, i, size(frames{1}));
        end
    end

    % Evaluate
    psnr_values = zeros(length(frames), 1);
    for i = 1:length(frames)
        psnr_values(i) = compute_psnr(frames{i}, recon_frames{i});
    end
    avg_psnr = mean(psnr_values);
    fprintf('Average PSNR: %.2f dB\n', avg_psnr);

    orig_size = sum(cellfun(@(x) numel(x) * 8, frames)); % Bits
    comp_size = sum(cellfun(@(x) huffman_size(x), compressed_data)); % Approximate
    ratio = orig_size / comp_size;
    fprintf('Compression Ratio: %.2f:1\n', ratio);
end

% Standalone Functions (non-nested)
function compressed = compress_frame_dcvc(frame, quant_step, prev_frame)
    if ~isempty(prev_frame)
        diff_frame = frame - prev_frame;
    else
        diff_frame = frame;
    end
    dct_frame = dct2(diff_frame);
    quant_frame = round(dct_frame / quant_step);
    compressed = {quant_frame, size(frame), quant_step};
end

function compressed = compress_frame_slimvc(frame, quant_step)
    compressed = compress_frame_dcvc(frame, quant_step, []);
end

function compressed = compress_frame_entropy(frame, quant_step, prev_frame)
    compressed_dcvc = compress_frame_dcvc(frame, quant_step, prev_frame); % Single output
    quant_frame = compressed_dcvc{1};
    frame_shape = compressed_dcvc{2};
    quant_step = compressed_dcvc{3}; % Same quant_step passed in, but reassigned for clarity
    flat_quant = int16(quant_frame(:)); % Flatten and convert
    [encoded, codebook] = huffman_encode(flat_quant);
    compressed = {encoded, frame_shape, quant_step, codebook};
end

function compressed = compress_frame_roi(frame, base_quant_step, prev_frame)
    edges = edge(frame, 'Canny');
    roi_mask = imdilate(edges, ones(5, 5));
    if ~isempty(prev_frame)
        diff_frame = frame - prev_frame;
    else
        diff_frame = frame;
    end
    dct_frame = dct2(diff_frame);
    quant_frame = zeros(size(dct_frame));
    quant_frame(roi_mask == 1) = round(dct_frame(roi_mask == 1) / (base_quant_step / 2));
    quant_frame(roi_mask == 0) = round(dct_frame(roi_mask == 0) / (base_quant_step * 2));
    compressed = {quant_frame, size(frame), base_quant_step, roi_mask};
end

function recon_frame = decompress_frame_dcvc(quant_frame, frame_shape, quant_step, prev_frame)
    dequant_frame = quant_frame * quant_step;
    recon_diff = idct2(dequant_frame);
    if ~isempty(prev_frame)
        recon_frame = prev_frame + recon_diff;
    else
        recon_frame = recon_diff;
    end
    recon_frame = uint8(min(max(recon_frame, 0), 255));
end

function recon_frame = decompress_frame_entropy(encoded, frame_shape, quant_step, codebook, prev_frame)
    flat_quant = huffman_decode(encoded, codebook);
    quant_frame = reshape(double(flat_quant), frame_shape);
    recon_frame = decompress_frame_dcvc(quant_frame, frame_shape, quant_step, prev_frame);
end

function recon_frame = decompress_frame_roi(quant_frame, frame_shape, base_quant_step, roi_mask, prev_frame)
    dequant_frame = zeros(frame_shape);
    dequant_frame(roi_mask == 1) = quant_frame(roi_mask == 1) * (base_quant_step / 2);
    dequant_frame(roi_mask == 0) = quant_frame(roi_mask == 0) * (base_quant_step * 2);
    recon_diff = idct2(dequant_frame);
    if ~isempty(prev_frame)
        recon_frame = prev_frame + recon_diff;
    else
        recon_frame = recon_diff;
    end
    recon_frame = uint8(min(max(recon_frame, 0), 255));
end

function recon_frame = decompress_frame_inr(key_frames, frame_idx, frame_shape)
    key_indices = cell2mat(keys(key_frames));
    if ismember(frame_idx, key_indices)
        recon_frame = key_frames(frame_idx);
        return;
    end
    for i = 1:length(key_indices)-1
        if key_indices(i) < frame_idx && frame_idx < key_indices(i+1)
            t = (frame_idx - key_indices(i)) / (key_indices(i+1) - key_indices(i));
            prev_frame = double(key_frames(key_indices(i)));
            next_frame = double(key_frames(key_indices(i+1)));
            recon_frame = uint8((1 - t) * prev_frame + t * next_frame);
            return;
        end
    end
    recon_frame = key_frames(key_indices(1)); % Fallback
end

function psnr_val = compute_psnr(original, reconstructed)
    mse = mean((double(original) - double(reconstructed)).^2, 'all');
    if mse == 0
        psnr_val = Inf;
    else
        psnr_val = 10 * log10((255^2) / mse);
    end
end

function [encoded, codebook] = huffman_encode(data)
    [symbols, ~, ic] = unique(data);
    freq = accumarray(ic, 1);
    if length(symbols) <= 1
        codebook = containers.Map(symbols(1), '0');
        encoded = repmat('0', 1, length(data));
        return;
    end
    nodes = [num2cell(freq), num2cell(symbols)];
    while length(nodes) > 1
        [~, idx] = sort(cell2mat(nodes(1, :)));
        nodes = nodes(:, idx);
        f1 = nodes{1, 1}; k1 = nodes{2, 1};
        f2 = nodes{1, 2}; k2 = nodes{2, 2};
        nodes(:, 1:2) = [];
        nodes = [[f1 + f2, {k1, k2}], nodes];
    end
    codebook = build_codes(nodes{2, 1});
    encoded = '';
    for i = 1:length(data)
        encoded = [encoded, codebook(data(i))];
    end
end

function codes = build_codes(tree, prefix)
    if ~iscell(tree)
        codes = containers.Map(tree, prefix);
        return;
    end
    codes = containers.Map();
    k1 = tree{1}; k2 = tree{2};
    codes = [codes; build_codes(k1, [prefix, '0'])];
    codes = [codes; build_codes(k2, [prefix, '1'])];
end

function decoded = huffman_decode(encoded, codebook)
    decodebook = containers.Map(values(codebook), keys(codebook));
    decoded = zeros(1, length(encoded), 'int16');
    current_code = '';
    idx = 1;
    for bit = encoded
        current_code = [current_code, bit];
        if isKey(decodebook, current_code)
            decoded(idx) = decodebook(current_code);
            idx = idx + 1;
            current_code = '';
        end
    end
    decoded = decoded(1:idx-1);
end

function size_bits = huffman_size(data)
    if ischar(data{1}) % Entropy
        size_bits = length(data{1}); % Bit string length
    else % DCVC, SlimVC, ROI
        size_bits = numel(data{1}) * 8; % Approximate
    end
end