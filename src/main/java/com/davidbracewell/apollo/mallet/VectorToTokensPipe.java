package com.davidbracewell.apollo.mallet;

import cc.mallet.pipe.Pipe;
import cc.mallet.types.Instance;
import cc.mallet.types.TokenSequence;
import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.ml.encoder.Encoder;
import com.davidbracewell.collection.Streams;
import com.davidbracewell.conversion.Cast;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class VectorToTokensPipe extends Pipe implements Serializable {
   private static final long serialVersionUID = 1L;
   private final Encoder encoder;

   public VectorToTokensPipe(Encoder encoder) {
      this.encoder = encoder;
   }

   @Override
   public Instance pipe(Instance inst) {
      NDArray vector = Cast.as(inst.getData());
      TokenSequence tokenSequence = new TokenSequence();
      Streams.asStream(vector.sparseIterator())
             .forEach(e -> tokenSequence.add(encoder.decode(e.getIndex()).toString()));
      inst.setData(tokenSequence);
      return inst;
   }


}//END OF VectorToTokensPipe
