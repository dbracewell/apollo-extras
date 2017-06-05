package com.davidbracewell.apollo.mallet;

import cc.mallet.pipe.Pipe;
import cc.mallet.types.Instance;
import cc.mallet.types.TokenSequence;
import com.davidbracewell.apollo.ml.sequence.Sequence;
import com.davidbracewell.conversion.Cast;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class SequenceToTokensPipe extends Pipe implements Serializable {
   private static final long serialVersionUID = 1L;


   @Override
   public Instance pipe(Instance inst) {
      TokenSequence tokenSequence = new TokenSequence();
      Sequence sequence = Cast.as(inst.getData());
      sequence.forEach(instance -> {
         if (instance.getFeatures().size() > 0) {
            tokenSequence.add(instance.getFeatures().get(0).getName());
         }
      });
      inst.setData(tokenSequence);
      return inst;
   }


}//END OF SequenceToTokensPipe
