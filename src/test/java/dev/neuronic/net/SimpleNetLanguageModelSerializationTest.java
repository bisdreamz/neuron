package dev.neuronic.net;

import dev.neuronic.net.activators.ReluActivator;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetLanguageModel;
import dev.neuronic.net.simple.SimpleNetTrainingConfig;
import dev.neuronic.net.layers.DenseLayer;
import dev.neuronic.net.layers.InputSequenceEmbeddingLayer;
import dev.neuronic.net.layers.GruLayer;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.outputs.SoftmaxCrossEntropyOutput;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test SimpleNetLanguageModel serialization and deserialization.
 */
class SimpleNetLanguageModelSerializationTest {
    
    @TempDir
    Path tempDir;
    
    @Test
    void testBasicLanguageModelSerialization() throws IOException {
        // Create a simple language model
        int sequenceLength = 5;
        int vocabularySize = 100;
        int embeddingSize = 32;
        int hiddenSize = 64;
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(sequenceLength)
            .layer(InputSequenceEmbeddingLayer.spec(sequenceLength, vocabularySize, embeddingSize, optimizer, WeightInitStrategy.XAVIER))
            .layer(GruLayer.specLast(hiddenSize, optimizer, WeightInitStrategy.XAVIER))
            .layer(DenseLayer.spec(hiddenSize, ReluActivator.INSTANCE, optimizer, WeightInitStrategy.HE))
            .output(SoftmaxCrossEntropyOutput.spec(vocabularySize, optimizer, WeightInitStrategy.XAVIER));
        
        SimpleNetLanguageModel originalModel = SimpleNet.ofLanguageModel(net);
        
        // Train with a few sequences
        String[][] sequences = {
            {"the", "quick", "brown", "fox", "jumps"},
            {"a", "lazy", "dog", "sleeps", "quietly"},
            {"the", "cat", "sat", "on", "the"},
            {"birds", "fly", "high", "in", "the"}
        };
        
        String[] nextWords = {"over", "today", "mat", "sky"};
        
        // Train the model
        for (int i = 0; i < sequences.length; i++) {
            originalModel.train(sequences[i], nextWords[i]);
        }
        
        // Make predictions before save
        String[] testSequence = {"the", "quick", "brown", "fox", "jumps"};
        String predictedBefore = originalModel.predictNext(testSequence);
        float[] probabilitiesBefore = originalModel.predictProbabilities(testSequence);
        String[] topKBefore = originalModel.predictTopK(testSequence, 5);
        
        // Test partial sequence prediction
        String[] partialSequence = {"the", "cat"};
        String predictedPartialBefore = originalModel.predictNextWithPadding(partialSequence);
        
        // Save to temp file
        Path modelFile = tempDir.resolve("language_model.bin");
        originalModel.save(modelFile);
        
        assertTrue(modelFile.toFile().exists(), "Model file should be created");
        assertTrue(modelFile.toFile().length() > 0, "Model file should not be empty");
        
        // Load from file
        SimpleNetLanguageModel loadedModel = SimpleNetLanguageModel.load(modelFile);
        
        // Verify predictions are identical after loading
        String predictedAfter = loadedModel.predictNext(testSequence);
        assertEquals(predictedBefore, predictedAfter, 
            "Predictions should be identical after loading");
        
        float[] probabilitiesAfter = loadedModel.predictProbabilities(testSequence);
        assertArrayEquals(probabilitiesBefore, probabilitiesAfter, 1e-3f,
            "Probability distributions should be similar after loading (optimizer state not preserved)");
        
        String[] topKAfter = loadedModel.predictTopK(testSequence, 5);
        assertArrayEquals(topKBefore, topKAfter,
            "Top-K predictions should be identical after loading");
        
        // Test partial sequence prediction after loading
        String predictedPartialAfter = loadedModel.predictNextWithPadding(partialSequence);
        assertEquals(predictedPartialBefore, predictedPartialAfter,
            "Partial sequence predictions should be identical after loading");
        
        // Verify vocabulary is preserved
        assertEquals(originalModel.getVocabularySize(), loadedModel.getVocabularySize(),
            "Vocabulary size should be preserved");
        
        // Check specific words
        for (String[] seq : sequences) {
            for (String word : seq) {
                assertTrue(loadedModel.hasWord(word),
                    "Word '" + word + "' should exist in loaded vocabulary");
            }
        }
        
        for (String word : nextWords) {
            assertTrue(loadedModel.hasWord(word),
                "Word '" + word + "' should exist in loaded vocabulary");
        }
    }
    
    @Test
    void testLanguageModelWithTrainingAndSerialization() throws IOException {
        // Create a more complex language model
        int sequenceLength = 10;
        int vocabularySize = 500;
        int embeddingSize = 64;
        int hiddenSize = 128;
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.0005f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(sequenceLength)
            .layer(InputSequenceEmbeddingLayer.spec(sequenceLength, vocabularySize, embeddingSize, optimizer, WeightInitStrategy.XAVIER))
            .layer(GruLayer.specLast(hiddenSize, optimizer, WeightInitStrategy.XAVIER))
            .layer(DenseLayer.spec(hiddenSize, ReluActivator.INSTANCE, optimizer, WeightInitStrategy.HE))
            .layer(DenseLayer.spec(64, ReluActivator.INSTANCE, optimizer, WeightInitStrategy.HE))
            .output(SoftmaxCrossEntropyOutput.spec(vocabularySize, optimizer, WeightInitStrategy.XAVIER));
        
        SimpleNetLanguageModel model = SimpleNet.ofLanguageModel(net);
        
        // Prepare training data
        List<String[]> sequences = new ArrayList<>();
        List<String> targets = new ArrayList<>();
        
        // Generate some synthetic training data
        String[] words = {"the", "a", "an", "cat", "dog", "bird", "runs", "walks", "flies",
                         "quick", "slow", "big", "small", "red", "blue", "green",
                         "in", "on", "under", "over", "through", "around"};
        
        // Create training sequences
        for (int i = 0; i < 50; i++) {
            String[] sequence = new String[sequenceLength];
            for (int j = 0; j < sequenceLength; j++) {
                sequence[j] = words[(i + j) % words.length];
            }
            sequences.add(sequence);
            targets.add(words[(i + sequenceLength) % words.length]);
        }
        
        // Train using bulk training
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
            .batchSize(10)
            .epochs(5)
            .verbosity(0)
            .build();
        
        model.trainBulk(sequences, targets, config);
        
        // Make predictions before save
        String[] testSeq = new String[sequenceLength];
        System.arraycopy(words, 0, testSeq, 0, sequenceLength);
        
        String prediction1 = model.predictNext(testSeq);
        String[] topK1 = model.predictTopK(testSeq, 3);
        
        // Save and load
        Path modelFile = tempDir.resolve("trained_language_model.bin");
        model.save(modelFile);
        
        SimpleNetLanguageModel loadedModel = SimpleNetLanguageModel.load(modelFile);
        
        // Verify predictions match
        String prediction2 = loadedModel.predictNext(testSeq);
        String[] topK2 = loadedModel.predictTopK(testSeq, 3);
        
        assertEquals(prediction1, prediction2, "Predictions should match after loading");
        assertArrayEquals(topK1, topK2, "Top-K predictions should match after loading");
        
        // Verify continued training works after loading
        loadedModel.train(testSeq, "test");
        
        // Should still be able to make predictions
        String prediction3 = loadedModel.predictNext(testSeq);
        assertNotNull(prediction3, "Should be able to predict after loading and training");
    }
    
    @Test
    void testEdgeCasesAndErrorHandling() throws IOException {
        // Create minimal language model
        int sequenceLength = 3;
        int vocabularySize = 10;
        
        AdamWOptimizer optimizer = new AdamWOptimizer(0.001f, 0.01f);
        
        NeuralNet net = NeuralNet.newBuilder()
            .input(sequenceLength)
            .layer(InputSequenceEmbeddingLayer.spec(sequenceLength, vocabularySize, 16, optimizer, WeightInitStrategy.XAVIER))
            .layer(GruLayer.specLast(32, optimizer, WeightInitStrategy.XAVIER))
            .output(SoftmaxCrossEntropyOutput.spec(vocabularySize, optimizer, WeightInitStrategy.XAVIER));
        
        SimpleNetLanguageModel model = SimpleNet.ofLanguageModel(net);
        
        // Train with minimal data including all words we'll use in tests
        model.train(new String[]{"a", "b", "c"}, "d");
        model.train(new String[]{"b", "c", "d"}, "e");
        model.train(new String[]{"c", "d", "e"}, "a");
        
        // Test padding functionality
        String[] shortSeq = {"a"};
        String padded = model.predictNextWithPadding(shortSeq);
        assertNotNull(padded, "Should handle short sequences with padding");
        
        // Test longer sequence (should truncate)
        String[] longSeq = {"a", "b", "c", "d", "e"};
        String truncated = model.predictNextWithPadding(longSeq);
        assertNotNull(truncated, "Should handle long sequences with truncation");
        
        // Save and verify padding still works after loading
        Path modelFile = tempDir.resolve("edge_case_model.bin");
        model.save(modelFile);
        
        SimpleNetLanguageModel loaded = SimpleNetLanguageModel.load(modelFile);
        
        // Test predictions are deterministic after loading
        String paddedAfter = loaded.predictNextWithPadding(shortSeq);
        assertEquals(padded, paddedAfter, "Padding behavior should be consistent after loading");
        
        String truncatedAfter = loaded.predictNextWithPadding(longSeq);
        assertEquals(truncated, truncatedAfter, "Truncation behavior should be consistent after loading");
        
        // Also verify that the vocabulary is properly preserved
        assertTrue(loaded.hasWord("a"));
        assertTrue(loaded.hasWord("b"));
        assertTrue(loaded.hasWord("c"));
        assertTrue(loaded.hasWord("d"));
        assertTrue(loaded.hasWord("e"));
    }
}