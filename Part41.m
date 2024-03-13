% Path to the directory containing the program files
data_dir = '/MATLAB Drive/';

% List all text files in the directory
file_names = dir(fullfile(data_dir, '*.txt'));
disp(file_names)

% Initialize a cell array to store all opcodes
all_opcodes = {};
opcode_counts = zeros(numel(file_names), 1);


% Combine all opcodes from the given programs
for i = 1:numel(file_names)
    file_path = fullfile(data_dir, file_names(i).name);
    %disp(file_names)
    file_opcodes = importdata(file_path);
    all_opcodes = [all_opcodes; file_opcodes];
end

% Count the occurrences of each opcode across all programs
opcode_counts = zeros(numel(all_opcodes), 1);
for i = 1:numel(file_names)
    file_path = fullfile(data_dir, file_names(i).name);
    file_opcodes = importdata(file_path);
    [~, idx] = ismember(file_opcodes, all_opcodes);
    opcode_counts(idx) = opcode_counts(idx) + 1;
end

% Determine stop words (opcodes that occur most frequently)
stop_words_threshold = 0.9 * numel(file_names);
stop_words_indices = opcode_counts >= stop_words_threshold;
stop_words = all_opcodes(stop_words_indices);

% Remove stop words from the list of opcodes
filtered_opcodes = setdiff(all_opcodes, stop_words);

% Create the dictionary D0 by removing duplicates
D0 = unique(filtered_opcodes);
document_vectors = zeros(numel(file_names), numel(D0));


% Represent each document as a vector using D0 (Vector Space Model)
for i = 1:numel(file_names)
    file_path = fullfile(data_dir, file_names(i).name);
    file_opcodes = importdata(file_path);
    document_vector = zeros(1, numel(D0));
    for j = 1:numel(D0)
        document_vector(j) = sum(strcmp(D0{j}, file_opcodes));
    end
    document_vectors(i, :) = document_vector;
end

% Term-Document Matrix (TDM)
term_document_matrix = document_vectors;

term_term_matrix = document_vectors' * document_vectors;

% Dot product kernel
dot_product_kernel = document_vectors * document_vectors';
%disp(dot_product_kernel)

% Polynomial kernel of degree 2
polynomial_kernel = (1 + dot_product_kernel).^2;
%disp(polynomial_kernel)
squared_euclidean_distance = 1 - dot_product_kernel;

% Apply K-means clustering to document vectors for dot product kernel
num_clusters = 2; % Choose the number of clusters as 3, 4 as asked in the assignemnt
%[idx_dp, centroids_dp] = kmeans(squared_euclidean_distance, num_clusters);

% Apply K-means clustering to document vectors for polynomial kernel
%[idx_poly, centroids_poly] = kmeans(polynomial_kernel, num_clusters);

% Apply Kernel Spectral Clustering
% Dot Product Kernel
W = exp(-squared_euclidean_distance ./ (2 * std(squared_euclidean_distance(:))^2)); % Similarity matrix
D = diag(sum(W, 2)); % Degree matrix
L = D - W; % Laplacian matrix
[v_dp, ~] = eigs(L, num_clusters, 'smallestabs'); % Eigen decomposition
v_dp = v_dp ./ sqrt(sum(v_dp.^2, 2)); % Normalize rows
[idx_dp, ~, ~] = kmeans(v_dp, num_clusters); % K-means clustering on rows of the normalized eigenvectors

% Polynomial Kernel of Degree 2
W_poly = exp(-polynomial_kernel ./ (2 * std(polynomial_kernel(:))^2)); % Similarity matrix
D_poly = diag(sum(W_poly, 2)); % Degree matrix
L_poly = D_poly - W_poly; % Laplacian matrix
[v_poly, ~] = eigs(L_poly, num_clusters, 'smallestabs'); % Eigen decomposition
v_poly = v_poly ./ sqrt(sum(v_poly.^2, 2)); % Normalize rows
[idx_poly, ~, ~] = kmeans(v_poly, num_clusters); % K-means clustering on rows of the normalized eigenvectors

% Display clustering results
disp('Clustering results using dot product kernel:');
disp(idx_dp);
disp('Clustering results using polynomial kernel (degree 2):');
disp(idx_poly);


class_labels = zeros(numel(file_names), 1);
for i = 1:numel(file_names)
    file_name = file_names(i).name;
    delimiter_index = strfind(file_name, '-');
    class_labels(i) = str2double(file_name(1:delimiter_index-1));
end

num_classes = length(unique(class_labels));

% Compute silhouette scores for dot product kernel clustering
silhouette_dp = silhouette(squared_euclidean_distance, idx_dp);

% Compute silhouette scores for polynomial kernel clustering
silhouette_poly = silhouette(polynomial_kernel, idx_poly);

% Display average silhouette scores
avg_silhouette_dp = mean(silhouette_dp);
avg_silhouette_poly = mean(silhouette_poly);

disp('Average Silhouette Score using dot product kernel clustering:');
disp(avg_silhouette_dp);
disp('Average Silhouette Score using polynomial kernel clustering:');
disp(avg_silhouette_poly);

% Plot silhouette histograms
figure;
subplot(1, 2, 1);
histogram(silhouette_dp, 'Normalization', 'probability');
title('Silhouette Score Distribution (Dot Product Kernel)');
xlabel('Silhouette Score');
ylabel('Frequency');
subplot(1, 2, 2);
histogram(silhouette_poly, 'Normalization', 'probability');
title('Silhouette Score Distribution (Polynomial Kernel)');
xlabel('Silhouette Score');
ylabel('Frequency');

% Check for presence of diagonal sub-matrices
if exist('class_labels', 'var') && ~isempty(class_labels)
  disp('** Analysis of Diagonal Sub-Matrices (assuming class labels in file names) **');

  % Extract sub-matrices for each class
  class_submatrices = cell(1, num_classes);
  for class_id = 1:num_classes
    class_indices = find(class_labels == class_id);
    class_submatrices{class_id} = dot_product_kernel(class_indices, class_indices);
  end

  % Analyze similarity within each sub-matrix
  for class_id = 1:num_classes
    class_submatrix = class_submatrices{class_id};

    % Statistical measures
    mean_similarity = mean(class_submatrix(:));
    std_dev_similarity = std(class_submatrix(:));

    disp(['  - Class ', num2str(class_id), ':']);
    disp(['      Mean similarity: ', num2str(mean_similarity)]);
    disp(['      Standard deviation of similarity: ', num2str(std_dev_similarity)]);
    
    figure;
    imagesc(class_submatrix);
    colorbar;
    title(['Heatmap of Class ', num2str(class_id), ' Sub-Matrix']);
    xlabel('Programs');
    ylabel('Programs');
    % - Cosine similarity
    cosine_similarity = diag(pdist2(document_vectors(class_indices,:), document_vectors(class_indices,:),'cosine'));
  end
else
  disp(' - Diagonal sub-matrices cannot be confirmed without class labels.');
end



