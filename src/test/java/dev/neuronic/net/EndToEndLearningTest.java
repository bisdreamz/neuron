package dev.neuronic.net;

import dev.neuronic.net.simple.SimpleNet;
import dev.neuronic.net.simple.SimpleNetLanguageModel;
import dev.neuronic.net.simple.SimpleNetTrainingConfig;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end tests to verify neural networks can learn simple patterns.
 */
public class EndToEndLearningTest {
    
    @Test
    public void testSimpleLinearLearning() {
        // Test if network can learn simple linear function: y = 2*x + 1
        NeuralNet net = NeuralNet.newBuilder()
            .input(1)
            .setDefaultOptimizer(new AdamWOptimizer(0.1f, 0.0f))  // Increased learning rate
            .layer(Layers.hiddenDenseRelu(8))
            .output(Layers.outputLinearRegression(1));
        
        // Training data: y = 2*x + 1
        float[][] inputs = {{1.0f}, {2.0f}, {3.0f}, {4.0f}, {5.0f}};
        float[][] targets = {{3.0f}, {5.0f}, {7.0f}, {9.0f}, {11.0f}};
        
        // Train for multiple epochs
        for (int epoch = 0; epoch < 100; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                net.train(inputs[i], targets[i]);
            }
        }
        
        // Test predictions
        float[] pred1 = net.predict(new float[]{1.0f});
        float[] pred2 = net.predict(new float[]{6.0f}); // Unseen data
        
        System.out.println("Linear learning test:");
        System.out.println("Input 1.0 -> Predicted: " + pred1[0] + " (expected ~3.0)");
        System.out.println("Input 6.0 -> Predicted: " + pred2[0] + " (expected ~13.0)");
        
        // Should learn approximately the correct function
        assertEquals(3.0f, pred1[0], 1.0f, "Should predict ~3 for input 1");
        assertEquals(13.0f, pred2[0], 2.0f, "Should predict ~13 for input 6");
    }
    
    @Test
    public void testLanguageModelPattern() {
        // Test if language model can learn simple deterministic pattern
        SimpleNetLanguageModel model = SimpleNet.ofLanguageModel(
            NeuralNet.newBuilder()
                .input(2) // 2-word context
                .setDefaultOptimizer(new AdamWOptimizer(0.1f, 0.0f))
                .layer(Layers.inputSequenceEmbedding(2, 10, 4))
                .layer(Layers.hiddenDenseRelu(8))
                .output(Layers.outputSoftmaxCrossEntropy(10))
        );
        
        // Simple deterministic pattern: "a b" -> "c", "b c" -> "a", "c a" -> "b" 
        String[][] sequences = {
            {"a", "b"}, {"a", "b"}, {"a", "b"},
            {"b", "c"}, {"b", "c"}, {"b", "c"},
            {"c", "a"}, {"c", "a"}, {"c", "a"}
        };
        String[] targets = {
            "c", "c", "c",
            "a", "a", "a", 
            "b", "b", "b"
        };
        
        // Train on simple pattern
        SimpleNetTrainingConfig config = SimpleNetTrainingConfig.builder()
            .batchSize(3)
            .epochs(50)
            .validationSplit(0.0f)
            .verbosity(0)
            .build();
            
        model.trainBulk(Arrays.asList(sequences), Arrays.asList(targets), config);
        
        // Test predictions 
        String pred1 = model.predictNext(new String[]{"a", "b"});
        String pred2 = model.predictNext(new String[]{"b", "c"});
        String pred3 = model.predictNext(new String[]{"c", "a"});
        
        System.out.println("\\nLanguage model pattern test:");
        System.out.println("'a b' -> '" + pred1 + "' (expected 'c')");
        System.out.println("'b c' -> '" + pred2 + "' (expected 'a')");
        System.out.println("'c a' -> '" + pred3 + "' (expected 'b')");
        
        // Should learn the deterministic pattern
        assertEquals("c", pred1, "Should predict 'c' after 'a b'");
        assertEquals("a", pred2, "Should predict 'a' after 'b c'");  
        assertEquals("b", pred3, "Should predict 'b' after 'c a'");
    }
    
    @Test
    public void testDifferentInputsDifferentOutputs() {
        // Verify that different inputs produce different outputs (basic sanity check)
        NeuralNet net = NeuralNet.newBuilder()
            .input(3)
            .setDefaultOptimizer(new AdamWOptimizer(0.1f, 0.0f))
            .layer(Layers.hiddenDenseRelu(5))
            .output(Layers.outputSoftmaxCrossEntropy(4));
        
        float[] input1 = {1.0f, 0.0f, 0.0f};
        float[] input2 = {0.0f, 1.0f, 0.0f};
        float[] input3 = {0.0f, 0.0f, 1.0f};
        
        float[] out1 = net.predict(input1);
        float[] out2 = net.predict(input2); 
        float[] out3 = net.predict(input3);
        
        System.out.println("\\nDifferent inputs test:");
        System.out.println("Input1: " + Arrays.toString(out1));
        System.out.println("Input2: " + Arrays.toString(out2));
        System.out.println("Input3: " + Arrays.toString(out3));
        
        // Outputs should be different
        assertFalse(Arrays.equals(out1, out2), "Different inputs should produce different outputs");
        assertFalse(Arrays.equals(out2, out3), "Different inputs should produce different outputs");
        assertFalse(Arrays.equals(out1, out3), "Different inputs should produce different outputs");
    }
}