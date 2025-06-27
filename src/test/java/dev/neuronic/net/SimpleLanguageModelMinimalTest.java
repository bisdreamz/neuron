package dev.neuronic.net;

import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetLanguageModel;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Minimal test to isolate the language model issue.
 */
public class SimpleLanguageModelMinimalTest {
    
    @Test
    public void testMinimalLearning() {
        // Super simple model - just embedding + linear output (no GRU)
        SimpleNetLanguageModel model = SimpleNet.ofLanguageModel(
            NeuralNet.newBuilder()
                .input(3)
                .setDefaultOptimizer(new AdamWOptimizer(0.1f, 0.0f)) // High LR, no weight decay
                .layer(Layers.inputSequenceEmbedding(3, 10, 8))
                .layer(Layers.hiddenDenseRelu(8)) // Just a simple dense layer
                .output(Layers.outputSoftmaxCrossEntropy(10))
        );
        
        // Train on just ONE simple example many times
        String[] sequence = {"a", "b", "c"};
        String target = "a";
        
        // Get initial prediction
        model.train(sequence, target); // Build vocab
        String initialPred = model.predictNext(sequence);
        float[] initialProbs = model.predictProbabilities(sequence);
        
        // Train many times on the same example
        for (int i = 0; i < 20; i++) {
            model.train(sequence, target);
        }
        
        // Get final prediction
        String finalPred = model.predictNext(sequence);
        float[] finalProbs = model.predictProbabilities(sequence);
        
        System.out.println("Initial prediction: " + initialPred);
        System.out.println("Final prediction: " + finalPred);
        System.out.println("Initial probs: " + Arrays.toString(initialProbs));
        System.out.println("Final probs: " + Arrays.toString(finalProbs));
        
        // After 20 training steps on "a b c" -> "a", it should predict "a"
        assertEquals("a", finalPred, "Model should learn to predict 'a' after 'a b c'");
    }
    
    @Test  
    public void testTwoSequenceLearning() {
        // Test with just two sequences to see if model can distinguish them
        SimpleNetLanguageModel model = SimpleNet.ofLanguageModel(
            NeuralNet.newBuilder()
                .input(3)
                .setDefaultOptimizer(new AdamWOptimizer(0.1f, 0.0f))
                .layer(Layers.inputSequenceEmbedding(3, 10, 8))
                .layer(Layers.hiddenDenseRelu(16))
                .output(Layers.outputSoftmaxCrossEntropy(10))
        );
        
        // Train on two distinct patterns
        String[] seq1 = {"a", "b", "c"};
        String target1 = "a";
        String[] seq2 = {"b", "c", "a"}; 
        String target2 = "b";
        
        // Alternate training on both examples
        for (int i = 0; i < 50; i++) {
            model.train(seq1, target1);
            model.train(seq2, target2);
        }
        
        // Test predictions
        String pred1 = model.predictNext(seq1);
        String pred2 = model.predictNext(seq2);
        float[] probs1 = model.predictProbabilities(seq1);
        float[] probs2 = model.predictProbabilities(seq2);
        
        System.out.println("After 'a b c': predicted='" + pred1 + "', expected='a'");
        System.out.println("After 'b c a': predicted='" + pred2 + "', expected='b'");
        System.out.println("Probs for 'a b c': " + Arrays.toString(probs1));
        System.out.println("Probs for 'b c a': " + Arrays.toString(probs2));
        
        assertEquals("a", pred1, "Should predict 'a' after 'a b c'");
        assertEquals("b", pred2, "Should predict 'b' after 'b c a'");
        
        // Ensure the probability distributions are actually different
        assertFalse(Arrays.equals(probs1, probs2), 
            "Different inputs should produce different probability distributions");
    }
}