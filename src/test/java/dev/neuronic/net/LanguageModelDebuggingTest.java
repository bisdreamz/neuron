package dev.neuronic.net;

import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetLanguageModel;
import dev.neuronic.net.simple.SimpleNetTrainingConfig;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Debugging tests to ensure the language model is actually learning different patterns
 * and not suffering from any hidden buffer corruption or identical output issues.
 */
public class LanguageModelDebuggingTest {
    
    @Test
    public void testActivationsChangeWithDifferentInputs() {
        // Create the same model as the failing test
        SimpleNetLanguageModel model = SimpleNet.ofLanguageModel(
            NeuralNet.newBuilder()
                .input(3)
                .setDefaultOptimizer(new AdamWOptimizer(0.01f, 0.0001f))
                .layer(Layers.inputSequenceEmbedding(3, 10, 8))
                .layer(Layers.hiddenGruLastNormalized(8))
                .output(Layers.outputSoftmaxCrossEntropy(10))
        );
        
        // Test that different inputs produce different outputs BEFORE training
        String[] input1 = {"a", "b", "c"};
        String[] input2 = {"b", "c", "a"};
        String[] input3 = {"c", "a", "b"};
        
        // Train once to build vocabulary
        model.train(input1, "a");
        
        float[] output1 = model.predictProbabilities(input1);
        float[] output2 = model.predictProbabilities(input2);
        float[] output3 = model.predictProbabilities(input3);
        
        System.out.println("Before training:");
        System.out.println("Input 'a b c' -> output: " + Arrays.toString(output1));
        System.out.println("Input 'b c a' -> output: " + Arrays.toString(output2));
        System.out.println("Input 'c a b' -> output: " + Arrays.toString(output3));
        
        // Verify outputs are different (not identical due to buffer corruption)
        assertFalse(Arrays.equals(output1, output2), 
            "Different inputs should produce different outputs (identical outputs suggest buffer corruption)");
        assertFalse(Arrays.equals(output1, output3), 
            "Different inputs should produce different outputs (identical outputs suggest buffer corruption)");
        assertFalse(Arrays.equals(output2, output3), 
            "Different inputs should produce different outputs (identical outputs suggest buffer corruption)");
    }
    
    @Test
    public void testActivationsChangeAfterTraining() {
        // Create model
        SimpleNetLanguageModel model = SimpleNet.ofLanguageModel(
            NeuralNet.newBuilder()
                .input(3)
                .setDefaultOptimizer(new AdamWOptimizer(0.01f, 0.0001f))
                .layer(Layers.inputSequenceEmbedding(3, 10, 8))
                .layer(Layers.hiddenGruLastNormalized(8))
                .output(Layers.outputSoftmaxCrossEntropy(10))
        );
        
        // Build vocabulary first
        String[] testInput = {"a", "b", "c"};
        model.train(testInput, "a");
        
        // Capture outputs before training
        float[] outputBefore = model.predictProbabilities(testInput);
        
        // Train on simple pattern
        List<String[]> sequences = Arrays.asList(
            new String[]{"a", "b", "c"},
            new String[]{"b", "c", "a"}, 
            new String[]{"c", "a", "b"}
        );
        List<String> targets = Arrays.asList("a", "b", "c");
        
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
            .batchSize(32)
            .epochs(10) // More epochs to ensure visible change
            .shuffle(false)
            .verbosity(0)
            .build();
            
        model.trainBulk(sequences, targets, config);
        
        // Capture outputs after training
        float[] outputAfter = model.predictProbabilities(testInput);
        
        System.out.println("Before training 'a b c': " + Arrays.toString(outputBefore));
        System.out.println("After training 'a b c':  " + Arrays.toString(outputAfter));
        
        // Verify that training actually changed the outputs
        assertFalse(Arrays.equals(outputBefore, outputAfter), 
            "Training should change the network outputs (identical outputs suggest the network isn't learning)");
        
        // Check that the difference is significant (not just tiny numerical differences)
        double totalDifference = 0;
        for (int i = 0; i < outputBefore.length; i++) {
            totalDifference += Math.abs(outputBefore[i] - outputAfter[i]);
        }
        assertTrue(totalDifference > 0.1, 
            "Training should cause significant output changes, got total difference: " + totalDifference);
    }
    
    
    
    @Test
    public void testTrainingPropagatesGradients() {
        // Test that training actually propagates gradients and changes weights
        SimpleNetLanguageModel model = SimpleNet.ofLanguageModel(
            NeuralNet.newBuilder()
                .input(3)
                .setDefaultOptimizer(new AdamWOptimizer(0.01f, 0.0001f))
                .layer(Layers.inputSequenceEmbedding(3, 10, 8))
                .layer(Layers.hiddenGruLastNormalized(8))
                .output(Layers.outputSoftmaxCrossEntropy(10))
        );
        
        // Train on a single example multiple times to see if weights change
        String[] sequence = {"a", "b", "c"};
        String target = "a";
        
        // Build vocabulary first
        model.train(sequence, target);
        
        // Get initial prediction
        String initialPrediction = model.predictNext(sequence);
        float[] initialOutput = model.predictProbabilities(sequence);
        
        // Train multiple times on the same example
        for (int i = 0; i < 10; i++) {
            model.train(sequence, target);
        }
        
        // Get final prediction
        String finalPrediction = model.predictNext(sequence);
        float[] finalOutput = model.predictProbabilities(sequence);
        
        System.out.println("Initial prediction for 'a b c': " + initialPrediction);
        System.out.println("Final prediction for 'a b c': " + finalPrediction);
        System.out.println("Initial output: " + Arrays.toString(initialOutput));
        System.out.println("Final output: " + Arrays.toString(finalOutput));
        
        // Verify that repeated training changed the outputs
        assertFalse(Arrays.equals(initialOutput, finalOutput),
            "Repeated training should change network outputs (identical outputs suggest gradients aren't propagating)");
        
        // Check for significant change
        double totalChange = 0;
        for (int i = 0; i < initialOutput.length; i++) {
            totalChange += Math.abs(initialOutput[i] - finalOutput[i]);
        }
        assertTrue(totalChange > 0.01, 
            "Training should cause measurable output changes, got total change: " + totalChange);
    }
    
    /**
     * Helper method to convert string sequences to token arrays
     */
    private float[] convertToTokens(String[] words) {
        // Simple mapping for our test vocabulary
        float[] tokens = new float[words.length];
        for (int i = 0; i < words.length; i++) {
            tokens[i] = (float) getTokenId(words[i]);
        }
        return tokens;
    }
    
    private int getTokenId(String word) {
        // Simple mapping for our test vocabulary
        switch (word) {
            case "a": return 1;
            case "b": return 2; 
            case "c": return 3;
            default: return 0; // UNK
        }
    }
}