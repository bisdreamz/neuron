package dev.neuronic.net.simple;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.SamplingConfig;
import dev.neuronic.net.layers.InputSequenceEmbeddingLayer;
import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.losses.CrossEntropyLoss;
import dev.neuronic.net.losses.Loss;
import dev.neuronic.net.serialization.SerializationConstants;
import dev.neuronic.net.training.BatchTrainer;
import dev.neuronic.net.training.TrainingCallback;
import dev.neuronic.net.training.EarlyStoppingCallback;
import dev.neuronic.net.training.ModelCheckpointCallback;
import dev.neuronic.net.training.VisualizationCallback;
import dev.neuronic.net.training.ProgressCallback;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Type-safe neural network wrapper for language modeling tasks.
 * 
 * <p><b>What it does:</b> Handles next-word prediction and text generation tasks
 * where both inputs and outputs are from the same vocabulary.
 * 
 * <p><b>Perfect for:</b>
 * <ul>
 *   <li><b>Text generation:</b> Generate coherent text continuations</li>
 *   <li><b>Language modeling:</b> Predict next words in sequences</li>
 *   <li><b>Autocomplete:</b> Suggest word completions</li>
 *   <li><b>Text understanding:</b> Model language patterns and structure</li>
 * </ul>
 * 
 * <p><b>Key Benefits:</b>
 * <ul>
 *   <li><b>Unified vocabulary:</b> Input and output share the same word space</li>
 *   <li><b>Automatic padding:</b> Handle variable-length prompts elegantly</li>
 *   <li><b>Clean API:</b> Simple methods for training and generation</li>
 *   <li><b>Efficient tokenization:</b> Reuses existing infrastructure</li>
 * </ul>
 * 
 * <p><b>Example - Text Generation:</b>
 * <pre>{@code
 * // Create language model
 * NeuralNet net = NeuralNet.newBuilder()
 *     .input(30) // sequence length
 *     .layer(Layers.inputSequenceEmbedding(30, 10000, 128))
 *     .layer(Layers.hiddenGruLast(256))
 *     .layer(Layers.hiddenDenseRelu(256))
 *     .output(Layers.outputSoftmaxCrossEntropy(10000));
 * 
 * SimpleNetLanguageModel lm = SimpleNet.ofLanguageModel(net);
 * 
 * // Train on sequences
 * lm.train(
 *     new String[]{"the", "cat", "sat", "on", "the"}, 
 *     "mat"
 * );
 * 
 * // Predict next word
 * String next = lm.predictNext(new String[]{"the", "cat", "sat", "on", "the"});
 * System.out.println("Next word: " + next); // "Next word: mat"
 * 
 * // Get top predictions
 * String[] top5 = lm.predictTopK(new String[]{"the", "cat"}, 5);
 * 
 * // Handle variable-length prompts with padding
 * String next2 = lm.predictNext(new String[]{"the", "cat"}); // Auto-pads
 * }</pre>
 * 
 * <p><b>Thread Safety:</b> All methods are thread-safe for concurrent training and prediction.
 */
public class SimpleNetLanguageModel extends SimpleNet<String> {
    
    private final InputSequenceEmbeddingLayer embeddingLayer;
    private final int sequenceLength;
    private final String paddingToken = "<pad>";
    private volatile boolean paddingInitialized = false;
    private volatile SamplingConfig samplingConfig = SamplingConfig.argmax();
    
    /**
     * Create a SimpleNetLanguageModel without output names (for backward compatibility and deserialization).
     * Package-private.
     */
    SimpleNetLanguageModel(NeuralNet underlyingNet) {
        this(underlyingNet, null, false);
    }
    
    /**
     * Create a SimpleNetLanguageModel.
     * Package-private - use SimpleNet.ofLanguageModel() instead.
     */
    SimpleNetLanguageModel(NeuralNet underlyingNet, Set<String> outputNames) {
        this(underlyingNet, outputNames, false);
    }
    
    /**
     * Private constructor with deserialization flag.
     */
    private SimpleNetLanguageModel(NeuralNet underlyingNet, Set<String> outputNames, boolean isDeserialized) {
        super(underlyingNet, outputNames);
        
        // Verify the network has InputSequenceEmbeddingLayer
        Layer inputLayer = underlyingNet.getInputLayer();
        if (!(inputLayer instanceof InputSequenceEmbeddingLayer)) {
            throw new IllegalArgumentException(
                "Language models require InputSequenceEmbeddingLayer as input layer");
        }
        
        this.embeddingLayer = (InputSequenceEmbeddingLayer) inputLayer;
        this.sequenceLength = embeddingLayer.getSequenceLength();
        
        // If deserialized, mark padding as already initialized to preserve vocabulary
        if (isDeserialized) {
            this.paddingInitialized = true;
        }
    }
    
    /**
     * Train the model with a single sequence and its next word.
     * 
     * @param sequence input sequence of words
     * @param nextWord the word that follows the sequence
     */
    public void train(String[] sequence, String nextWord) {
        if (sequence.length != sequenceLength) {
            throw new IllegalArgumentException(String.format(
                "Sequence length must be %d, got %d", sequenceLength, sequence.length));
        }
        
        // Ensure padding token is in vocabulary on first train
        ensurePaddingInitialized();
        
        // Tokenize input sequence
        float[] tokenIds = tokenizeSequence(sequence);
        
        // Create target (one-hot vector for next word)
        int targetTokenId = embeddingLayer.getTokenId(nextWord);
        float[] target = createOneHotTarget(targetTokenId);
        
        // Train
        underlyingNet.train(tokenIds, target);
    }
    
    /**
     * Train the model with multiple sequences using bulk training.
     * 
     * @param sequences list of input sequences
     * @param nextWords list of next words (parallel to sequences)
     * @param config training configuration
     * @return training result with metrics
     */
    public SimpleNetTrainingResult trainBulkSequences(List<String[]> sequences, List<String> nextWords,
                                           SimpleNetTrainingConfig config) {
        return trainBulkSequences(sequences, nextWords, config, null);
    }
    
    /**
     * Train the model with multiple sequences using bulk training with custom callbacks.
     * 
     * @param sequences list of input sequences
     * @param nextWords list of next words (parallel to sequences)
     * @param config training configuration
     * @param customCallbacks additional callbacks to use during training
     * @return training result with metrics
     */
    public SimpleNetTrainingResult trainBulkSequences(List<String[]> sequences, List<String> nextWords,
                                           SimpleNetTrainingConfig config,
                                           List<TrainingCallback> customCallbacks) {
        if (sequences.size() != nextWords.size()) {
            throw new IllegalArgumentException("Sequences and nextWords must have same size");
        }
        
        // Ensure padding token is in vocabulary
        ensurePaddingInitialized();
        
        // Tokenize all sequences (potentially in parallel)
        float[][] tokenizedInputs = tokenizeSequencesBulk(sequences);
        float[][] encodedTargets = new float[nextWords.size()][];
        
        // Encode targets
        for (int i = 0; i < nextWords.size(); i++) {
            int targetTokenId = embeddingLayer.getTokenId(nextWords.get(i));
            encodedTargets[i] = createOneHotTarget(targetTokenId);
        }
        
        // Create BatchTrainer first so we can access its stop flag
        BatchTrainer trainer = new BatchTrainer(underlyingNet, CrossEntropyLoss.INSTANCE, config.getBatchConfig());
        
        // Build callbacks list
        List<TrainingCallback> callbacks = new ArrayList<>();
        
        if (config.isEarlyStoppingEnabled()) {
            // For language models, monitor val_loss instead of val_accuracy
            callbacks.add(new EarlyStoppingCallback(
                config.getEarlyStoppingPatience(),
                config.getEarlyStoppingMinDelta(),
                trainer.getStopFlag(),  // Use trainer's stop flag
                "val_loss",            // Monitor validation loss for language models
                false                   // Don't restore best weights
            ));
        }
        
        if (config.isCheckpointingEnabled()) {
            callbacks.add(new ModelCheckpointCallback.WithModel(
                underlyingNet,
                config.getCheckpointPath(),
                "val_accuracy",
                config.isCheckpointOnlyBest(),
                0
            ));
        }
        
        if (config.isVisualizationEnabled()) {
            callbacks.add(new VisualizationCallback(config.getVisualizationPath()));
        }
        
        // Add custom callbacks if provided
        if (customCallbacks != null)
            callbacks.addAll(customCallbacks);
        
        // Train using BatchTrainer directly
        long startTime = System.currentTimeMillis();
        
        // Add language model specific progress callback if verbosity is enabled
        if (config.getBatchConfig().verbosity > 0) {
            boolean detailed = config.getBatchConfig().verbosity == 2;
            trainer.withCallback(ProgressCallback.forLanguageModel(detailed));
        }
        
        // Add all callbacks
        for (TrainingCallback callback : callbacks) {
            trainer.withCallback(callback);
        }
        
        BatchTrainer.TrainingResult batchResult = trainer.fit(
            tokenizedInputs, encodedTargets);
        long trainingTime = System.currentTimeMillis() - startTime;
        
        return new SimpleNetTrainingResult(
            batchResult,
            trainingTime,
            batchResult.getMetrics().getEpochCount()
        );
    }
    
    /**
     * Train the model with pre-split train and validation data.
     * 
     * @param trainSequences training input sequences
     * @param trainNextWords training target words
     * @param valSequences validation input sequences
     * @param valNextWords validation target words
     * @param config training configuration
     * @return training result with metrics
     */
    public SimpleNetTrainingResult trainBulk(List<String[]> trainSequences, List<String> trainNextWords,
                                           List<String[]> valSequences, List<String> valNextWords,
                                           SimpleNetTrainingConfig config) {
        return trainBulk(trainSequences, trainNextWords, valSequences, valNextWords, config, null);
    }
    
    /**
     * Train the model with pre-split data and custom callbacks.
     */
    public SimpleNetTrainingResult trainBulk(List<String[]> trainSequences, List<String> trainNextWords,
                                           List<String[]> valSequences, List<String> valNextWords,
                                           SimpleNetTrainingConfig config,
                                           List<TrainingCallback> customCallbacks) {
        // Validate inputs
        if (trainSequences.size() != trainNextWords.size())
            throw new IllegalArgumentException("Training sequences and nextWords must have same size");
        
        if (valSequences != null && valNextWords != null && valSequences.size() != valNextWords.size())
            throw new IllegalArgumentException("Validation sequences and nextWords must have same size");
        
        // Ensure padding token is in vocabulary
        ensurePaddingInitialized();
        
        // Tokenize training data
        float[][] trainInputs = tokenizeSequencesBulk(trainSequences);
        float[][] trainTargets = new float[trainNextWords.size()][];
        
        for (int i = 0; i < trainNextWords.size(); i++) {
            int targetTokenId = embeddingLayer.getTokenId(trainNextWords.get(i));
            trainTargets[i] = createOneHotTarget(targetTokenId);
        }
        
        // Tokenize validation data if provided
        float[][] valInputs = null;
        float[][] valTargets = null;
        
        if (valSequences != null && valNextWords != null) {
            valInputs = tokenizeSequencesBulk(valSequences);
            valTargets = new float[valNextWords.size()][];
            
            for (int i = 0; i < valNextWords.size(); i++) {
                int targetTokenId = embeddingLayer.getTokenId(valNextWords.get(i));
                valTargets[i] = createOneHotTarget(targetTokenId);
            }
        }
        
        // Create BatchTrainer first so we can access its stop flag
        BatchTrainer trainer = new BatchTrainer(underlyingNet, CrossEntropyLoss.INSTANCE, config.getBatchConfig());
        
        // Build callbacks
        List<TrainingCallback> callbacks = new ArrayList<>();
        
        if (config.isEarlyStoppingEnabled()) {
            // For language models, monitor val_loss instead of val_accuracy
            callbacks.add(new EarlyStoppingCallback(
                config.getEarlyStoppingPatience(),
                config.getEarlyStoppingMinDelta(),
                trainer.getStopFlag(),  // Use trainer's stop flag
                "val_loss",            // Monitor validation loss for language models
                false                   // Don't restore best weights
            ));
        }
        
        if (config.isCheckpointingEnabled()) {
            callbacks.add(new ModelCheckpointCallback.WithModel(
                underlyingNet,
                config.getCheckpointPath(),
                "val_accuracy",
                config.isCheckpointOnlyBest(),
                0
            ));
        }
        
        if (config.isVisualizationEnabled()) {
            callbacks.add(new VisualizationCallback(config.getVisualizationPath()));
        }
        
        if (customCallbacks != null)
            callbacks.addAll(customCallbacks);
        
        // Train using BatchTrainer directly with pre-split data
        long startTime = System.currentTimeMillis();
        
        // Add language model specific progress callback if verbosity is enabled
        if (config.getBatchConfig().verbosity > 0) {
            boolean detailed = config.getBatchConfig().verbosity == 2;
            trainer.withCallback(ProgressCallback.forLanguageModel(detailed));
        }
        
        // Add all callbacks
        for (TrainingCallback callback : callbacks) {
            trainer.withCallback(callback);
        }
        
        BatchTrainer.TrainingResult batchResult = trainer.fit(
            trainInputs, trainTargets, valInputs, valTargets);
        long trainingTime = System.currentTimeMillis() - startTime;
        
        return new SimpleNetTrainingResult(
            batchResult,
            trainingTime,
            batchResult.getMetrics().getEpochCount()
        );
    }
    
    /**
     * Predict the next word given a sequence.
     * 
     * @param sequence input sequence (must be sequenceLength)
     * @return predicted next word
     */
    public String predictNext(String[] sequence) {
        if (sequence.length != sequenceLength) {
            throw new IllegalArgumentException(String.format(
                "Sequence length must be %d, got %d", sequenceLength, sequence.length));
        }
        
        // Tokenize and predict
        float[] tokenIds = tokenizeSequence(sequence);
        
        // Apply sampling strategy using new prediction methods
        int predictedTokenId = switch (samplingConfig.getStrategy()) {
            case ARGMAX -> (int) underlyingNet.predictArgmax(tokenIds);
            case TEMPERATURE -> (int) underlyingNet.predictWithTemperature(tokenIds, samplingConfig.getTemperature());
            case TOP_K -> (int) underlyingNet.predictSampleTopK(tokenIds, samplingConfig.getK(), samplingConfig.getTemperature());
            case TOP_P -> (int) underlyingNet.predictSampleTopP(tokenIds, samplingConfig.getP(), samplingConfig.getTemperature());
        };
        
        // Convert back to word
        return embeddingLayer.getWord(predictedTokenId);
    }
    
    
    
    /**
     * Pad or truncate a sequence to match the model's expected sequence length.
     * 
     * @param sequence input sequence (any length)
     * @return padded/truncated sequence of length sequenceLength
     */
    public String[] padSequence(String[] sequence) {
        ensurePaddingInitialized();
        
        if (sequence.length == sequenceLength) {
            return sequence;
        } else if (sequence.length > sequenceLength) {
            // Truncate - take the last sequenceLength words
            String[] truncated = new String[sequenceLength];
            System.arraycopy(sequence, sequence.length - sequenceLength, 
                           truncated, 0, sequenceLength);
            return truncated;
        } else {
            // Pad with padding tokens
            String[] padded = new String[sequenceLength];
            int padCount = sequenceLength - sequence.length;
            
            // Fill with padding tokens
            Arrays.fill(padded, 0, padCount, paddingToken);
            
            // Copy actual sequence
            System.arraycopy(sequence, 0, padded, padCount, sequence.length);
            
            return padded;
        }
    }
    
    /**
     * Predict the next word given a partial sequence (auto-pads).
     * 
     * @param partialSequence input sequence (can be shorter than sequenceLength)
     * @return predicted next word
     */
    public String predictNextWithPadding(String[] partialSequence) {
        // Ensure padding token is in vocabulary
        ensurePaddingInitialized();
        
        if (partialSequence.length > sequenceLength) {
            // Take the last sequenceLength words
            String[] truncated = new String[sequenceLength];
            System.arraycopy(partialSequence, partialSequence.length - sequenceLength, 
                           truncated, 0, sequenceLength);
            return predictNext(truncated);
        } else if (partialSequence.length < sequenceLength) {
            // Pad with padding tokens
            String[] padded = new String[sequenceLength];
            int padCount = sequenceLength - partialSequence.length;
            
            // Fill with padding tokens - use a token that will map to UNK if not in vocabulary
            Arrays.fill(padded, 0, padCount, paddingToken);
            
            // Copy actual sequence
            System.arraycopy(partialSequence, 0, padded, padCount, partialSequence.length);
            
            return predictNext(padded);
        } else {
            return predictNext(partialSequence);
        }
    }
    
    /**
     * Get the top K predicted next words.
     * 
     * @param sequence input sequence
     * @param k number of top predictions to return
     * @return array of top K predicted words
     */
    public String[] predictTopK(String[] sequence, int k) {
        if (sequence.length != sequenceLength) {
            throw new IllegalArgumentException(String.format(
                "Sequence length must be %d, got %d", sequenceLength, sequence.length));
        }
        
        // Tokenize and get top K predictions
        float[] tokenIds = tokenizeSequence(sequence);
        float[] topKIndices = underlyingNet.predictTopK(tokenIds, k);
        
        // Convert to words
        String[] topWords = new String[topKIndices.length];
        for (int i = 0; i < topKIndices.length; i++) {
            topWords[i] = embeddingLayer.getWord((int) topKIndices[i]);
        }
        
        return topWords;
    }
    
    /**
     * Get the probability distribution over next words.
     * 
     * @param sequence input sequence
     * @return probability for each word in vocabulary
     */
    public float[] predictProbabilities(String[] sequence) {
        if (sequence.length != sequenceLength) {
            throw new IllegalArgumentException(String.format(
                "Sequence length must be %d, got %d", sequenceLength, sequence.length));
        }
        
        float[] tokenIds = tokenizeSequence(sequence);
        return underlyingNet.predict(tokenIds);
    }
    
    /**
     * Get the vocabulary size.
     */
    public int getVocabularySize() {
        return embeddingLayer.getVocabularySize();
    }
    
    /**
     * Check if a word is in the vocabulary.
     */
    public boolean hasWord(String word) {
        return embeddingLayer.hasWord(word);
    }
    
    
    // ===============================
    // PRIVATE HELPER METHODS
    // ===============================
    
    private void ensurePaddingInitialized() {
        if (!paddingInitialized) {
            synchronized (this) {
                if (!paddingInitialized) {
                    // Add padding token to vocabulary
                    embeddingLayer.getTokenId(paddingToken);
                    paddingInitialized = true;
                }
            }
        }
    }
    
    private float[] tokenizeSequence(String[] sequence) {
        float[] tokenIds = new float[sequence.length];
        for (int i = 0; i < sequence.length; i++) {
            tokenIds[i] = embeddingLayer.getTokenId(sequence[i]);
        }
        return tokenIds;
    }
    
    private float[][] tokenizeSequencesBulk(List<String[]> sequences) {
        float[][] tokenized = new float[sequences.size()][];
        
        // Use parallel processing for large datasets
        if (sequences.size() > 100) {
            int numThreads = Runtime.getRuntime().availableProcessors();
            ExecutorService executor = Executors.newFixedThreadPool(numThreads);
            int chunkSize = (sequences.size() + numThreads - 1) / numThreads;
            
            List<java.util.concurrent.Future<?>> futures = new ArrayList<>();
            
            for (int t = 0; t < numThreads; t++) {
                final int startIdx = t * chunkSize;
                final int endIdx = Math.min(startIdx + chunkSize, sequences.size());
                
                futures.add(executor.submit(() -> {
                    for (int i = startIdx; i < endIdx; i++) {
                        tokenized[i] = tokenizeSequence(sequences.get(i));
                    }
                }));
            }
            
            // Wait for completion
            for (java.util.concurrent.Future<?> future : futures) {
                try {
                    future.get();
                } catch (Exception e) {
                    throw new RuntimeException("Error during parallel tokenization", e);
                }
            }
            
            executor.shutdown();
        } else {
            // Sequential processing for small datasets
            for (int i = 0; i < sequences.size(); i++) {
                tokenized[i] = tokenizeSequence(sequences.get(i));
            }
        }
        
        return tokenized;
    }
    
    private float[] createOneHotTarget(int tokenId) {
        Layer outputLayer = underlyingNet.getOutputLayer();
        float[] target = new float[outputLayer.getOutputSize()];
        
        // Ensure token ID is within bounds
        if (tokenId >= 0 && tokenId < target.length) {
            target[tokenId] = 1.0f;
        } else {
            // Default to UNK token
            target[0] = 1.0f;
        }
        
        return target;
    }
    
    private int[] getTopKIndices(float[] values, int k) {
        // Create index-value pairs
        Integer[] indices = new Integer[values.length];
        for (int i = 0; i < values.length; i++) {
            indices[i] = i;
        }
        
        // Sort by value (descending)
        Arrays.sort(indices, (a, b) -> Float.compare(values[b], values[a]));
        
        // Return top K
        int actualK = Math.min(k, values.length);
        int[] topK = new int[actualK];
        for (int i = 0; i < actualK; i++) {
            topK[i] = indices[i];
        }
        
        return topK;
    }
    
    // ===============================
    // SAVE/LOAD METHODS
    // ===============================
    
    /**
     * Save this language model to a file.
     * 
     * @param path file path to save to
     * @throws IOException if save fails
     */
    public void save(Path path) throws IOException {
        try (DataOutputStream out = new DataOutputStream(java.nio.file.Files.newOutputStream(path))) {
            writeTo(out, SerializationConstants.CURRENT_VERSION);
        }
    }
    
    /**
     * Load a language model from a file.
     * 
     * @param path file path to load from
     * @return loaded language model
     * @throws IOException if load fails
     */
    public static SimpleNetLanguageModel load(Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(java.nio.file.Files.newInputStream(path))) {
            return deserialize(in, SerializationConstants.CURRENT_VERSION);
        }
    }
    
    // Serialization implementation
    
    @Override
    public void writeTo(DataOutputStream out, int version) throws IOException {
        // Write type identifier
        out.writeInt(getTypeId());
        
        // Write the underlying neural network
        underlyingNet.writeTo(out, version);
    }
    
    @Override
    public void readFrom(DataInputStream in, int version) throws IOException {
        throw new UnsupportedOperationException("Use load() static method instead");
    }
    
    /**
     * Deserialize a SimpleNetLanguageModel from stream.
     */
    public static SimpleNetLanguageModel deserialize(DataInputStream in, int version) throws IOException {
        // Read and verify type identifier
        int typeId = in.readInt();
        if (typeId != (SerializationConstants.TYPE_NEURAL_NET + 100)) {
            String actualType = getTypeNameFromId(typeId);
            throw new IOException("Type mismatch: This file contains a " + actualType + 
                " model, but you're trying to load it as SimpleNetLanguageModel. " +
                "Use " + actualType + ".load() instead.");
        }
        
        // Read the underlying neural network
        NeuralNet net = NeuralNet.deserialize(in, version);
        
        // Create the language model with deserialization flag
        SimpleNetLanguageModel model = new SimpleNetLanguageModel(net, null, true);
        return model;
    }
    
    @Override
    public int getSerializedSize(int version) {
        // Type ID (4 bytes) + neural network size
        return 4 + underlyingNet.getSerializedSize(version);
    }
    
    @Override
    public int getTypeId() {
        // Using a high number to avoid conflicts with existing type IDs
        return SerializationConstants.TYPE_NEURAL_NET + 100;
    }
    
    private static String getTypeNameFromId(int typeId) {
        switch (typeId) {
            case SerializationConstants.TYPE_SIMPLE_NET:
                return "SimpleNetInt";
            case SerializationConstants.TYPE_SIMPLE_NET_FLOAT:
                return "SimpleNetFloat";
            case SerializationConstants.TYPE_SIMPLE_NET_STRING:
                return "SimpleNetString";
            case SerializationConstants.TYPE_NEURAL_NET + 100:
                return "SimpleNetLanguageModel";
            default:
                return "Unknown type (ID: " + typeId + ")";
        }
    }
    
    /**
     * Set the sampling configuration for text generation.
     * 
     * <p><b>Examples:</b>
     * <pre>{@code
     * // For deterministic generation (default)
     * model.setSamplingConfig(SamplingConfig.argmax());
     * 
     * // For creative text generation
     * model.setSamplingConfig(SamplingConfig.temperature(1.2f));
     * 
     * // For balanced generation with vocabulary restriction
     * model.setSamplingConfig(SamplingConfig.topK(40, 0.8f));
     * 
     * // For high-quality generation
     * model.setSamplingConfig(SamplingConfig.topP(0.9f, 0.8f));
     * }</pre>
     * 
     * @param config the sampling configuration to use
     */
    public void setSamplingConfig(SamplingConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("Sampling config cannot be null");
        }
        this.samplingConfig = config;
    }
    
    /**
     * Get the current sampling configuration.
     */
    public SamplingConfig getSamplingConfig() {
        return samplingConfig;
    }
    
    // ===============================
    // SIMPLENET BASE CLASS METHODS
    // ===============================
    
    @Override
    protected void trainInternal(Object input, float[] targets) {
        // Ensure padding token is in vocabulary on first train
        ensurePaddingInitialized();
        
        // Input should be a string array
        if (!(input instanceof String[])) {
            throw new IllegalArgumentException("Language model input must be String[]");
        }
        
        String[] sequence = (String[]) input;
        if (sequence.length != sequenceLength) {
            throw new IllegalArgumentException(String.format(
                "Sequence length must be %d, got %d", sequenceLength, sequence.length));
        }
        
        // Tokenize input sequence
        float[] tokenIds = tokenizeSequence(sequence);
        
        // Train
        underlyingNet.train(tokenIds, targets);
    }
    
    @Override
    protected float[] predictInternal(Object input) {
        // Ensure padding token is in vocabulary
        ensurePaddingInitialized();
        
        // Input should be a string array
        if (!(input instanceof String[])) {
            throw new IllegalArgumentException("Language model input must be String[]");
        }
        
        String[] sequence = (String[]) input;
        
        // Handle short sequences by padding
        if (sequence.length < sequenceLength) {
            sequence = padSequence(sequence);
        } else if (sequence.length > sequenceLength) {
            // Take the last N tokens
            String[] truncated = new String[sequenceLength];
            System.arraycopy(sequence, sequence.length - sequenceLength, 
                           truncated, 0, sequenceLength);
            sequence = truncated;
        }
        
        // Tokenize input
        float[] tokenIds = tokenizeSequence(sequence);
        
        // Get predictions
        return underlyingNet.predict(tokenIds);
    }
    
    @Override
    public SimpleNetTrainingResult trainBulk(List<?> inputs, List<String> targets,
                                            SimpleNetTrainingConfig config) {
        if (inputs.isEmpty()) {
            return trainWithEncodedData(new float[0][], new float[0][], config);
        }
        
        Object firstInput = inputs.get(0);
        
        if (firstInput instanceof String[]) {
            // This is the expected input type for language models
            // Cast is safe because we checked the type
            @SuppressWarnings("unchecked")
            List<String[]> sequences = (List<String[]>) inputs;
            return trainBulkSequences(sequences, targets, config);
        } else {
            throw new UnsupportedOperationException(
                "Language models require String[] inputs. Got: " + 
                firstInput.getClass().getSimpleName() + 
                ". Use String[] sequences for language model training.");
        }
    }
    
    @Override
    protected Loss getLossFunction() {
        return CrossEntropyLoss.INSTANCE;
    }
    
    @Override
    protected String getCheckpointMonitorMetric() {
        return "val_accuracy";  // For language models, monitor validation accuracy
    }
    
    @Override
    protected Object predictFromArray(float[] input) {
        // Language models don't use raw float arrays as input
        throw new UnsupportedOperationException(
            "Language models require String[] inputs. Use predictNext(String[]) instead.");
    }
    
    @Override
    protected Object predictFromMap(Map<String, Object> input) {
        // Language models don't use Map inputs
        throw new UnsupportedOperationException(
            "Language models require String[] inputs. Use predictNext(String[]) instead.");
    }
    
    @Override
    protected float[][] encodeTargets(List<String> targets) {
        // Ensure padding token is in vocabulary
        ensurePaddingInitialized();
        
        // Convert string targets to one-hot encoded vectors
        float[][] encoded = new float[targets.size()][];
        for (int i = 0; i < targets.size(); i++) {
            int tokenId = embeddingLayer.getTokenId(targets.get(i));
            encoded[i] = createOneHotTarget(tokenId);
        }
        return encoded;
    }
}