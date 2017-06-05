package com.davidbracewell.apollo.mallet;

import cc.mallet.pipe.Pipe;
import cc.mallet.types.Instance;
import cc.mallet.types.TokenSequence;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.conversion.Cast;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class VectorToTokensPipe extends Pipe implements Serializable {
   private static final long serialVersionUID = 1L;


   @Override
   public Instance pipe(Instance inst) {
      FeatureVector vector = Cast.as(inst.getData());
      TokenSequence tokenSequence = new TokenSequence();
      vector.decodedFeatureStream().forEach(f -> tokenSequence.add(f.getKey().toString()));
      inst.setData(tokenSequence);
      return inst;
   }


}//END OF VectorToTokensPipe
