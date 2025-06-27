package dev.neuronic.net.layers;

import dev.neuronic.net.layers.Layer.LayerContext;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test that DropoutLayer correctly handles dynamic input sizes.
 */
public class DropoutLayerDynamicSizeTest {
    
    @Test
    public void testDynamicDropoutAdaptsToInputSize() {
        // Create dynamic dropout layer
        DropoutLayer dropout = new DropoutLayer(0.5f);
        assertEquals(-1, dropout.getOutputSize());
        
        // Test with different input sizes
        float[] input128 = new float[128];
        float[] input256 = new float[256];
        float[] input2560 = new float[2560];
        
        // Should handle all sizes without throwing
        LayerContext ctx1 = dropout.forward(input128);
        assertEquals(128, ctx1.outputs().length);
        
        LayerContext ctx2 = dropout.forward(input256);
        assertEquals(256, ctx2.outputs().length);
        
        LayerContext ctx3 = dropout.forward(input2560);
        assertEquals(2560, ctx3.outputs().length);
    }
    
    @Test
    public void testFixedSizeDropoutValidatesInput() {
        // Create fixed-size dropout layer
        DropoutLayer dropout = new DropoutLayer(0.5f, 128);
        assertEquals(128, dropout.getOutputSize());
        
        // Should accept correct size
        float[] correctInput = new float[128];
        LayerContext ctx = dropout.forward(correctInput);
        assertEquals(128, ctx.outputs().length);
        
        // Should reject incorrect size
        float[] wrongInput = new float[256];
        assertThrows(IllegalArgumentException.class, () -> dropout.forward(wrongInput));
    }
    
    @Test
    public void testDropoutAfterVariableSizeGRU() {
        // Simulate GRU output with different sizes
        DropoutLayer dropout = new DropoutLayer(0.3f);
        
        // GRU all timesteps: sequence_length * hidden_size
        float[] gruAllTimesteps = new float[20 * 128]; // 2560
        LayerContext ctx1 = dropout.forward(gruAllTimesteps);
        assertEquals(2560, ctx1.outputs().length);
        
        // GRU last timestep: just hidden_size  
        float[] gruLastTimestep = new float[128];
        LayerContext ctx2 = dropout.forward(gruLastTimestep);
        assertEquals(128, ctx2.outputs().length);
    }
    
    @Test
    public void testBackwardAlsoHandlesDynamicSizes() {
        DropoutLayer dropout = new DropoutLayer(0.5f);
        
        // Forward with one size
        float[] input = new float[256];
        for (int i = 0; i < input.length; i++) {
            input[i] = 1.0f;
        }
        LayerContext ctx = dropout.forward(input);
        
        // Backward should handle same size
        float[] upstreamGrad = new float[256];
        for (int i = 0; i < upstreamGrad.length; i++) {
            upstreamGrad[i] = 1.0f;
        }
        
        LayerContext[] stack = {ctx};
        float[] downstreamGrad = dropout.backward(stack, 0, upstreamGrad);
        assertEquals(256, downstreamGrad.length);
    }
}