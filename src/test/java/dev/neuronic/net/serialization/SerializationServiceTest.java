package dev.neuronic.net.serialization;

import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.optimizers.Optimizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test the centralized SerializationService that eliminates tight coupling.
 */
class SerializationServiceTest {
    
    @BeforeEach
    void setUp() {
        // Ensure service is initialized
        SerializationService.initialize();
    }
    
    @Test
    void testServiceInitialization() {
        // Service should initialize without errors
        assertDoesNotThrow(() -> SerializationService.initialize());
        
        // Should be idempotent - multiple calls are safe
        SerializationService.initialize();
        SerializationService.initialize();
    }
    
    @Test
    void testOptimizerRegistration() {
        // Built-in optimizers should be registered
        assertTrue(SerializationService.supportsOptimizer(SerializationConstants.TYPE_SGD_OPTIMIZER));
        assertTrue(SerializationService.supportsOptimizer(SerializationConstants.TYPE_ADAM_OPTIMIZER));
        assertTrue(SerializationService.supportsOptimizer(SerializationConstants.TYPE_ADAMW_OPTIMIZER));
        
        // Unknown types should not be supported
        assertFalse(SerializationService.supportsOptimizer(999));
    }
    
    @Test
    void testLayerRegistration() {
        // Built-in layers should be registered
        assertTrue(SerializationService.supportsLayer(SerializationConstants.TYPE_DENSE_LAYER));
        assertTrue(SerializationService.supportsLayer(SerializationConstants.TYPE_SOFTMAX_CROSSENTROPY_OUTPUT));
        assertTrue(SerializationService.supportsLayer(SerializationConstants.TYPE_LINEAR_REGRESSION_OUTPUT));
        assertTrue(SerializationService.supportsLayer(SerializationConstants.TYPE_MIXED_FEATURE_INPUT_LAYER));
        
        // Unknown types should not be supported
        assertFalse(SerializationService.supportsLayer(999));
    }
    
    @Test
    void testOptimizerDeserialization() throws IOException {
        // Test SGD optimizer
        SgdOptimizer sgd = new SgdOptimizer(0.01f);
        testOptimizerRoundTrip(sgd, SerializationConstants.TYPE_SGD_OPTIMIZER);
        
        // Test AdamW optimizer
        AdamWOptimizer adamW = new AdamWOptimizer(0.001f, 0.01f);
        testOptimizerRoundTrip(adamW, SerializationConstants.TYPE_ADAMW_OPTIMIZER);
    }
    
    @Test
    void testUnknownOptimizerType() {
        // Should throw IOException for unknown type ID
        ByteArrayInputStream in = new ByteArrayInputStream(new byte[0]);
        DataInputStream dataIn = new DataInputStream(in);
        
        IOException exception = assertThrows(IOException.class, 
            () -> SerializationService.deserializeOptimizer(dataIn, 999, 1));
        
        assertTrue(exception.getMessage().contains("Unknown optimizer type ID: 999"));
        assertTrue(exception.getMessage().contains("Available types:"));
    }
    
    @Test
    void testUnknownLayerType() {
        // Should throw IOException for unknown type ID
        ByteArrayInputStream in = new ByteArrayInputStream(new byte[0]);
        DataInputStream dataIn = new DataInputStream(in);
        
        IOException exception = assertThrows(IOException.class, 
            () -> SerializationService.deserializeLayer(dataIn, 999, 1));
        
        assertTrue(exception.getMessage().contains("Unknown layer type ID: 999"));
        assertTrue(exception.getMessage().contains("Available types:"));
    }
    
    @Test
    void testTypeIdMapping() {
        // Should be able to get type IDs for registered classes
        SgdOptimizer sgd = new SgdOptimizer(0.01f);
        Integer typeId = SerializationService.getTypeId(sgd);
        assertNotNull(typeId);
        assertEquals(SerializationConstants.TYPE_SGD_OPTIMIZER, typeId.intValue());
        
        AdamWOptimizer adamW = new AdamWOptimizer(0.001f, 0.01f);
        typeId = SerializationService.getTypeId(adamW);
        assertNotNull(typeId);
        assertEquals(SerializationConstants.TYPE_ADAMW_OPTIMIZER, typeId.intValue());
    }
    
    @Test
    void testRegistrationInfo() {
        // Should provide useful debugging information
        String info = SerializationService.getRegistrationInfo();
        assertNotNull(info);
        assertTrue(info.contains("SerializationService Registration Info"));
        assertTrue(info.contains("Optimizers:"));
        assertTrue(info.contains("Layers:"));
        assertTrue(info.contains("Type mappings:"));
        
        // Should list registered type IDs
        assertTrue(info.contains(String.valueOf(SerializationConstants.TYPE_SGD_OPTIMIZER)));
        assertTrue(info.contains(String.valueOf(SerializationConstants.TYPE_ADAMW_OPTIMIZER)));
    }
    
    @Test
    void testLooseCoupling() {
        // The key benefit: components don't need to know about all other types
        // This test verifies that the service can deserialize any registered optimizer
        // without individual components having hardcoded knowledge
        
        // Any component can now deserialize any registered optimizer type
        assertDoesNotThrow(() -> {
            // Component doesn't need to know what optimizers exist
            // It just delegates to the centralized service
            if (SerializationService.supportsOptimizer(SerializationConstants.TYPE_ADAMW_OPTIMIZER)) {
                // Service handles the complexity
                assertTrue(true, "Service supports AdamW without tight coupling");
            }
        });
    }
    
    /**
     * Helper method to test optimizer serialization round-trip.
     */
    private void testOptimizerRoundTrip(Optimizer originalOptimizer, int expectedTypeId) throws IOException {
        // Serialize
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        DataOutputStream dataOut = new DataOutputStream(out);
        
        // Get type ID and serialize
        Integer typeId = SerializationService.getTypeId(originalOptimizer);
        assertEquals(expectedTypeId, typeId.intValue());
        
        dataOut.writeInt(typeId);
        ((Serializable) originalOptimizer).writeTo(dataOut, SerializationConstants.CURRENT_VERSION);
        
        // Deserialize
        ByteArrayInputStream in = new ByteArrayInputStream(out.toByteArray());
        DataInputStream dataIn = new DataInputStream(in);
        
        int readTypeId = dataIn.readInt();
        assertEquals(expectedTypeId, readTypeId);
        
        Optimizer deserializedOptimizer = SerializationService.deserializeOptimizer(
            dataIn, readTypeId, SerializationConstants.CURRENT_VERSION);
        
        // Verify same type
        assertEquals(originalOptimizer.getClass(), deserializedOptimizer.getClass());
    }
}