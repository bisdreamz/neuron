package dev.neuronic.net;

import dev.neuronic.net.layers.InputSequenceEmbeddingLayer;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetString;
import dev.neuronic.net.simple.SimpleNetTrainingConfig;
import dev.neuronic.net.simple.SimpleNetTrainingResult;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test that InputSequenceEmbeddingLayer works properly with bulk training through SimpleNetString.
 */
public class InputSequenceEmbeddingBulkTrainingTest {
    
    @Test
    public void testBulkTrainingWithStringSequences() {
        // Create a simple language model with sequence embedding
        int sequenceLength = 5;
        int vocabSize = 100;
        int embeddingDim = 16;
        int hiddenSize = 32;
        
        NeuralNet model = NeuralNet.newBuilder()
                .input(sequenceLength)
                .setDefaultOptimizer(new AdamWOptimizer(0.01f, 0.01f))
                .layer(Layers.inputSequenceEmbedding(sequenceLength, vocabSize, embeddingDim))
                .layer(Layers.hiddenGruLast(hiddenSize))
                .output(Layers.outputSoftmaxCrossEntropy(3)); // 3 classes
        
        SimpleNetString classifier = SimpleNet.ofStringClassification(model);
        
        // Prepare training data - sequences of words
        List<Object> sequences = new ArrayList<>();
        List<String> labels = new ArrayList<>();
        
        // Add some sample sequences
        sequences.add(new String[]{"the", "quick", "brown", "fox", "jumps"});
        labels.add("animal");
        
        sequences.add(new String[]{"hello", "world", "from", "java", "code"});
        labels.add("greeting");
        
        sequences.add(new String[]{"neural", "network", "training", "is", "fun"});
        labels.add("tech");
        
        sequences.add(new String[]{"the", "fox", "is", "very", "quick"}); // Reuses some words
        labels.add("animal");
        
        // Test single example training first
        classifier.train(sequences.get(0), labels.get(0));
        
        // Now test bulk training
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
                .batchSize(2)
                .epochs(10)
                .build();
        
        SimpleNetTrainingResult result = classifier.trainBulk(sequences, labels, config);
        
        assertNotNull(result);
        assertTrue(result.getEpochsTrained() > 0);
        
        // Test prediction on a new sequence
        String[] testSequence = {"the", "brown", "fox", "runs", "fast"};
        String prediction = classifier.predictString(testSequence);
        
        // Debug output
        System.out.println("Test sequence: " + String.join(" ", testSequence));
        System.out.println("Predicted: " + prediction);
        System.out.println("Available labels: " + labels);
        System.out.println("Final accuracy: " + result.getFinalAccuracy());
        
        // Should predict one of our classes
        assertTrue(labels.contains(prediction), 
            "Prediction '" + prediction + "' not in labels: " + labels);
        
        // Verify vocabulary was built properly
        InputSequenceEmbeddingLayer embeddingLayer = (InputSequenceEmbeddingLayer) model.getInputLayer();
        assertTrue(embeddingLayer.hasWord("the"));
        assertTrue(embeddingLayer.hasWord("fox"));
        assertTrue(embeddingLayer.hasWord("hello"));
        
        // Check vocabulary size
        assertTrue(embeddingLayer.getVocabularySize() > 10); // Should have built vocab from all sequences
    }
    
    @Test
    public void testMixedVocabularyHandling() {
        // Test that unknown words are handled properly
        int sequenceLength = 3;
        int vocabSize = 10; // Small vocab to test overflow
        
        NeuralNet model = NeuralNet.newBuilder()
                .input(sequenceLength)
                .setDefaultOptimizer(new AdamWOptimizer(0.01f, 0.01f))
                .layer(Layers.inputSequenceEmbedding(sequenceLength, vocabSize, 8))
                .layer(Layers.hiddenDenseRelu(16))
                .output(Layers.outputSoftmaxCrossEntropy(2));
        
        SimpleNetString classifier = SimpleNet.ofStringClassification(model);
        
        // Train with more words than vocab size
        List<Object> sequences = new ArrayList<>();
        List<String> labels = new ArrayList<>();
        
        for (int i = 0; i < 15; i++) {
            sequences.add(new String[]{"word" + i, "common", "text"});
            labels.add(i % 2 == 0 ? "even" : "odd");
        }
        
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
                .batchSize(5)
                .epochs(5)
                .build();
        
        // Should handle vocabulary overflow gracefully
        SimpleNetTrainingResult result = classifier.trainBulk(sequences, labels, config);
        assertNotNull(result);
        
        // Test with unknown words
        String[] testSequence = {"unknown", "word", "here"};
        String prediction = classifier.predictString(testSequence);
        assertTrue(prediction.equals("even") || prediction.equals("odd"));
    }
}