#ifndef _WHILEBLOCKRTC_H
#define _WHILEBLOCKRTC_H

#include "RTC_LineRTC.hh"
#include "RTC_BlockRTC.hh"
#include "RTC_ObjectRTC.hh"
#include "RTC_TokenizerRTC.hh"

#include <string>
#include <map>

namespace PG_RuntimeCompiler {

/**
 * A WhileBlock represents a block of code within a while-loop. 
 * WhileBlock extends Block because it is a Block.
 */

class WhileBlock : public Block
{
 public:

  /**
   * Constructor -> The constructor contructs the parent Block, finds and 
   *                creates the condition for the while loop, and tells parent 
   *                to create its sub statements. 
   *
   * @param vars  - A map of already active variables
   * @param lines - The array of strings that represent the lines of the code. 
   * @param errs  - A string containing the errors that have been generated by
   *                the compiling of lines. If errs is not empty, then the 
   *                program has not compiled succesfully
   */  
  WhileBlock(std::map<std::string, Variable*> vars, Tokenizer& lines, 
             std::string& err);

  /**
   * Destructor -> The destructor deletes _condition
   */
  ~WhileBlock();

  /**
   * execute -> This method executes this WhileBlock. _condition is evaluated 
   *            and the loop enters if _condition is true. After each loop,
   *             _condition is re-evaluated to determine if we loop again.
   */  
  Value* execute();

 private:

  Line* _condition; /**!< The code that gets checked at the beginning of each 
                     *    loop. If _condition returns false, the looping ends.
                     */
};

}

#endif
