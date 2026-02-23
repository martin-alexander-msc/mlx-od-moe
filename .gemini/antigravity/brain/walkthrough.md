# Walkthrough

1. Inspected `gguf.quants` API and verified built-in dequantizer availability.
2. Probed the user GGUF blob and confirmed dequantized arrays differ in axis order from metadata.
3. Updated converter tensor read path to dequantize and align to metadata shape.
4. Retained packed MoE extraction and explicit expert extraction compatibility.
5. Added tests for reversed-axis alignment and Q4 dequantization output shape.
6. Updated README note to reflect Q4 conversion support path.
