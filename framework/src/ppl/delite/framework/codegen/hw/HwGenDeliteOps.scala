package ppl.delite.framework.codegen.hw

import ppl.delite.framework.codegen.delite.DeliteKernelCodegen
import scala.virtualization.lms.internal._
import scala.virtualization.lms.common._

// All IR nodes, GenericGenDeliteOps
import ppl.delite.framework.ops._
import ppl.delite.framework.Config
import ppl.delite.framework.datastructures._

// Analysis passes
import ppl.delite.framework.analysis.PrimitiveReduceAnalysis
import ppl.delite.framework.analysis.MetaPipelineAnalysis
import ppl.delite.framework.analysis.DotPrintAnalysis

import java.io.File
import java.io.PrintWriter
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.Map
import scala.collection.mutable.Set
import scala.collection.mutable.Stack


trait HwGenDeliteOps extends HwCodegen with GenericGenDeliteOps
{
  // FIXME: This needs to be changed - temporarily put this here just to make things compile
//  val IR: DeliteOpsExp with LoopsFatExp with ArrayOpsExp with StringOpsExp
  import IR._

  // New stuff from merge with wip-master (some need to be filled in?)
  def emitHeapMark(): Unit = {}
  def emitHeapReset(result: List[String]): Unit = {}
  def emitAbstractFatLoopFooter(syms: List[Sym[Any]], rhs: AbstractFatLoop): Unit = {}
  def emitAbstractFatLoopHeader(syms: List[Sym[Any]], rhs: AbstractFatLoop): Unit = {}
  def syncType(actType: String): String = "??????"
  def emitWorkLaunch(kernelName: String, rSym: String, allocSym: String, syncSym: String): Unit = {}

  def isPrimitiveReduce(elem: DeliteReduceElem[_]) = {
    val m = elem.mA.toString
    m match {
      case "Int" => true
      case "Float" => true
      case "Double" => true
      case _ => false
    }
  }

  def emitScalarReduceFSM[A](rfunc: Block[A]) = {

    def emitSimpleReduceFn(rfunc: Block[A]) = {
      val analysis = new PrimitiveReduceAnalysis {val IR: HwGenDeliteOps.this.IR.type = HwGenDeliteOps.this.IR}
      val map = analysis.run(rfunc).toList
      if (map.size == 0 || map.size > 1) {
        sys.error(s"Primitive reduce function has more than 1 primitive op! $map")
      }
      map(0)._2 match {
        case DIntPlus(lhs, rhs) =>
          stream.println("counterFF.next <== sm_d + counter;")
        case DIntMinus(lhs, rhs) =>
          stream.println("counterFF.next <== sm_d - counter;")
        case DIntTimes(lhs, rhs) =>
          stream.println("counterFF.next <== sm_d * counter;")
        case DIntDivide(lhs, rhs) =>
          stream.println("counterFF.next <== sm_d / counter;")
        case DLessThan(lhs, rhs) =>
          stream.println("counterFF.next <== sm_d < counter;")
        case DGreaterThan(lhs, rhs) =>
          stream.println("counterFF.next <== sm_d > counter;")
        case _ =>
          sys.error(s"Unknown primitive op ${map(0)._2}")
      }
    }

    stream.println("package engine;")
    stream.println("import com.maxeler.maxcompiler.v2.kernelcompiler.KernelLib;")
    stream.println("import com.maxeler.maxcompiler.v2.statemachine.DFEsmInput;")
    stream.println("import com.maxeler.maxcompiler.v2.statemachine.DFEsmOutput;")
    stream.println("import com.maxeler.maxcompiler.v2.statemachine.DFEsmStateEnum;")
    stream.println("import com.maxeler.maxcompiler.v2.statemachine.DFEsmStateValue;")
    stream.println("import com.maxeler.maxcompiler.v2.statemachine.kernel.KernelStateMachine;")
    stream.println("import com.maxeler.maxcompiler.v2.statemachine.types.DFEsmValueType;")

    stream.println(s"class ScalarReduceFSM_${getBlockResult(rfunc)} extends KernelStateMachine {")
    stream.println("// State IO")
    stream.println("  private final DFEsmInput sm_d;")
    stream.println("  private final DFEsmInput en;")
    stream.println("  private final DFEsmOutput sm_q;")
    stream.println("  // Accumulator")
    stream.println("    private final DFEsmStateValue counterFF;")

   stream.println(s"public ScalarReduceFSM_${getBlockResult(rfunc)} (KernelLib owner) {")
   stream.println("   super(owner);")
   stream.println("   DFEsmValueType ffType = dfeUInt(32);")
   stream.println("   DFEsmValueType wireType = dfeBool();")
   stream.println("   sm_d = io.input(\"sm_d\", ffType);")
   stream.println("   en = io.input(\"en\", wireType);")
   stream.println("   sm_q = io.output(\"sm_q\", ffType);")
   stream.println(" counterFF = state.value(ffType, 0);")
   stream.println("}")

   stream.println("@Override")
   stream.println("protected void outputFunction() {")
   stream.println("sm_q <== counterFF;")
   stream.println("}")

   stream.println("@Override")
   stream.println("protected void nextState() {")
   stream.println("IF (en) {")
   emitSimpleReduceFn(rfunc)
   stream.println(" }")
   stream.println("}")
   stream.println("}")
  }

  override def emitNode(sym: Sym[Any], rhs: Def[Any]) = {
    curSym.push(sym)
    rhs match {
      case op: AbstractLoop[_] =>
        stream.println(s"// $sym is an AbstractLoop")
          seenLoops += sym
      case op: AbstractLoopNest[_] =>
        stream.println(s"// $sym is an AbstractLoopNest")
        // 1. Emit FSM if the body needs one
        // 2. Instantiate FSM
        // 3. Emit Counterchain
        //      - Connect 'en' from s0_done
        // 4. if (FSM) emit FF:
        //      - keep track of boundSym -> prefix
        //      - Emit FF chain instantiation
        // 5. Emit Double buffers - Need to track this
        //    in the metadata analysis!
        // 6. Emit each state - keep appropriate prefix/suffix
        //    in a map before calling emitBlock. See how 'i'
        //    and 'jburst' are used in S1 and S2 in example design

        val loopName = quote(sym)
        // 1. Emit FSM if the body needs one
        val metapipelineAnalysis = new MetaPipelineAnalysis {val IR: HwGenDeliteOps.this.IR.type = HwGenDeliteOps.this.IR}
        val smStages = metapipelineAnalysis.run(sym, op.body, seenLoops).asInstanceOf[List[Sym[Any]]]
        println(s"smStages: $smStages")
        val needsSM = !smStages.isEmpty
        if (needsSM) {
          // Add all stages to the dblBufMap
          smStages.foreach(dblBufMap.add(_))

          val smInputs = smStages.map(x => s"${quote(x)}_done")
          val smOutputs = smStages.map(x => s"${quote(x)}_en")

          stream.println(s"// Loop $loopName needs a SM with the following spec:")
          stream.println(s"""
            // inputs: $smInputs
            // outputs: $smOutputs
            // numIter: ${op.sizes.map(quote(_)).reduce(_+"*"+_)}
            """)

          stream.println(s"""DFEVar ${loopName}_numIter = ${op.sizes.map(quote(_)).reduce(_+"*"+_)};""")

          stream.println(s"""
            SMIO ${loopName}_sm = addStateMachine(\"${loopName}_sm\", new ${loopName}_StateMachine(this));
            ${loopName}_sm.connectInput(\"sm_en\", ${loopName}_en);
            ${loopName}_done <== stream.offset(${loopName}_sm.getOutput(\"sm_done\"),-1);
            sm.connectInput(\"sm_numIter\", ${loopName}_numIter);
            """)

          for (idx <- 0 until smInputs.size) {
            val i: String = smInputs(idx)
            val o: String = smOutputs(idx)
            stream.println(s"""
              DFEVar ${i} = dfeBool().newInstance(this);
              ${loopName}_sm.connectInput(\"s${idx}_done\", ${i});
              DFEVar $o = sm.getOutput(\"s${idx}_en\") & ${loopName}_en;""")
          }

          val fsmWriter = new PrintWriter(s"${bDir}/${loopName}_StateMachine.${fileExtension}")
          withStream(fsmWriter) {
            emitSM(loopName, smStages.map(x =>quote(x)))
          }
          fsmWriter.close()
        }

        // 3. Emit counterchain
        val counterEn = if (needsSM) s"${quote(smStages(0))}_done" else s"${loopName}_done";
        stream.println(s"CounterChain chain_$loopName = control.count.makeCounterChain($counterEn);")
        for (i <- 0 until op.vs.size) {
          val v = quote(op.vs(i))
          val maxLen = quote(op.sizes(i))
          val stride = quote(op.strides(i))
          stream.println(s"""DFEVar $v = chain_$loopName.addCounter($maxLen, $stride);""")
        }

        // 4. Emit FF chain to propagate counters if needed
        if (needsSM) {
          for (i <- 0 until smStages.size) {
            for (j <- 0 until op.vs.size) {
              val s = quote(smStages(i))
              val v = quote(op.vs(j))
              val storage = if (i == 0) "DFEVar" else "FFLib"
              val storageName = s"${s}_${v}"
              arbitPrefixMap += (smStages(i), op.vs(j)) -> s"$s"
              val prevName = if (i == 0) "" else if (i == 1) s"${quote(smStages(i-1))}_${v}" else s"${quote(smStages(i-1))}_${v}.q"
              if (i > 0) {
                baseKernelLibStream.println(s"protected static $storage ${storageName}_ff;")
              }
              baseKernelLibStream.println(s"protected static DFEVar ${storageName};")

              if (i == 0) {
                stream.println(s"$storageName = $v;")
              } else {
                stream.println(s"""${storageName}_ff = new FFLib(owner, \"$storageName\", ${s}_done, $prevName);""")
                stream.println(s"""${storageName} = ${storageName}_ff.q;""")
              }
            }
          }
        }

        stream.println("// Begin body emission")
        op.body match {
          case elem: DeliteCollectElem[_,_,_] =>
            aliasMap(getBlockResult(elem.buf.alloc)) = aliasMap.getOrElse(sym,sym)
            aliasMap(elem.buf.allocVal) = aliasMap.getOrElse(sym,sym)
            aliasMap(elem.buf.sV) = getMemorySize(sym)
            aliasMap(elem.buf.eV) = aliasMap.getOrElse(getBlockResult(elem.func),getBlockResult(elem.func))
            stream.println("// CollectElem alloc")
            emitBlock(elem.buf.alloc)
            stream.println("// CollectElem func")
            emitBlock(elem.func)
            stream.println("// CollectElem update")
            emitBlock(elem.buf.update)

          case elem: DeliteReduceElem[_] =>
          case elem: DeliteTileElem[_,_,_] =>
          case _ => sys.error(s"Unknown loop body ${op.body}")
        }
        stream.println("// End body emission")
        stream.println(s"// Arbit. prefix map = $arbitPrefixMap");
        seenLoops += sym

      case _ =>
        super.emitNode(sym, rhs)
    }
    curSym.pop
  }

  override def emitFatNode(symList: List[Sym[Any]], rhs: FatDef) = rhs match {
    case op: AbstractFatLoop =>
      if (Config.debugCodegen) {
        println(s"[codegen] HwGenDeliteOps::emitFatNode::AbstractFatLoop, op = $op, symList = $symList")
      }

      val loopName = symList.map(quote(_)).reduce(_+_)
      val symBodyTuple = symList.zip(op.body)
      println(s"symBodyTuple: $symBodyTuple")
      // Create alias table (boundsSym -> resultSym)
      // Output buffer must have the same name as the loop symbol
      val prevAliasMap = aliasMap
      symBodyTuple.foreach {
        case (sym, elem:DeliteCollectElem[_,_,_]) =>
          aliasMap(getBlockResult(elem.buf.alloc)) = aliasMap.getOrElse(sym,sym)
          aliasMap(elem.buf.allocVal) = aliasMap.getOrElse(sym,sym)
          aliasMap(elem.buf.sV) = op.size
          aliasMap(elem.buf.eV) = aliasMap.getOrElse(getBlockResult(elem.func),getBlockResult(elem.func))
        case (sym, elem: DeliteReduceElem[_]) =>
//          aliasMap(elem.rV._1) = getBlockResult(elem.func)
//          aliasMap(elem.rV._2) = sym
//          aliasMap(getBlockResult(elem.rFunc)) = sym
        case _ =>
          throw new Exception("Not handled yet")
      }

      // MetaPipeline analysis - does this loop need a controlling FSM?
      // What should it look like?
      val metapipelineAnalysis = new MetaPipelineAnalysis {val IR: HwGenDeliteOps.this.IR.type = HwGenDeliteOps.this.IR}
      val dotPrintAnalysis = new DotPrintAnalysis {val IR: HwGenDeliteOps.this.IR.type = HwGenDeliteOps.this.IR}


      val bodySMInfo = symBodyTuple.map { t =>
        val bodyMetadata = metapipelineAnalysis.run(t._1, t._2, seenLoops)
        (t._1, bodyMetadata)
      }.groupBy { _._1 }
      .mapValues{ x => List(x(0)._2).flatten.asInstanceOf[List[Sym[Any]]] }
      println(s"bodySMInfo: $bodySMInfo")

      // FSM v/s CounterChain decision here:
      //here If bodySMInfo has atleast one body requiring a SM, emit FSM
      // Else emitCounter
      // Emit counter - this should be agnostic of loop body
      // The counter stride will change based on the parallelism factor
      // Keeping it at 1 for now
      val needsSM = !bodySMInfo.mapValues(_.isEmpty).values.toList.reduce(_&_)

      if (needsSM) {
        if (bodySMInfo.size > 1) {
          sys.error(s"bodySMInfo = $bodySMInfo \nFused loop needs more than one SM, which isn't handled now!")
        }

        val smStages = bodySMInfo.values.toList(0).reverse
        val smIter = quote(op.size)
        val smPipeline = false  // TODO: Set this the output of a Config flag

        val smInputs = smStages.map(x => s"done_${quote(x)}")
        val smOutputs = smStages.map(x => s"en_${quote(x)}")

        stream.println(s"// Loop $loopName needs a SM with the following spec:")
        stream.println(s"""
          // inputs: $smInputs
          // outputs:  $smOutputs
          // count: $smIter
          // pipeline: $smPipeline
          """)

        stream.println(s"""
          SMIO ${loopName}_sm = addStateMachine(\"${loopName}_sm\", new ${loopName}_StateMachine(this, ${smIter}));
          ${loopName}_sm.connectInput(\"sm_en\", en_$loopName); // TODO:Verify if 'en' is right
          DFEVar done_$loopName = ${loopName}_sm.getOutput(\"sm_done\");
          DFEVar ${quote(op.v)} = ${loopName}_sm.getOutput(\"sm_count\");""")

          for (idx <- 0 until smInputs.size) {
            val i: String = smInputs(idx)
            val o: String = smOutputs(idx)
            stream.println(s"""
              ${i} = dfeBool().newInstance(this);
              ${loopName}_sm.connectInput(\"s${idx}_done\", ${i});
              DFEVar $o = sm.getOutput(\"s${idx}_en\") & en_$loopName; // TODO: Verify if 'en' is right""")
          }

        val fsmWriter = new PrintWriter(s"${bDir}/${loopName}_StateMachine.${fileExtension}")
        withStream(fsmWriter) {
          emitSM(loopName, smStages.map(x =>quote(x)))
        }
        fsmWriter.close()

      } else {
        stream.println(s"// Loop $loopName does NOT need a SM")
        stream.println(s"CounterChain ${loopName}_chain = control.count.makeCounterChain(en_${loopName});")
        stream.println(s"DFEVar ${quote(op.v)} = ${loopName}_chain.addCounter(${quote(op.size)}, 1);")
        stream.println(s"done_${loopName} <== stream.offset(${loopName}_chain.getCounterWrap(${quote(op.v)}), -1);")

      }

      // In case stuff is fused, emit functions only once
      // Note that counters should be emitted before this
      // as function bodies emitted here will most certainly depend on that
      emitMultiLoopFuncs(op, symList)

      // Generate code for each body
      symBodyTuple.foreach { t =>
        curSym.push(t._1)
        t match {
          case (sym, elem: DeliteCollectElem[_,_,_]) =>
            // TODO: Check first if op.size is a const, else assert
  //          stream.println(s"Count.Params ${quote(op.v)}_params = control.count.makeParams(addrBits)")
  //          stream.println(s"      .withEnable(en)")
  //          stream.println(s"      .withMax(${quote(op.size)});")
  //
  //          stream.println(s"Count.Counter ${quote(op.v)} = control.count.makeCounter(${quote(op.v)}_params);")
  //          stream.println(s"done <== stream.offset(${quote(op.v)}.getWrap(), -1);")

            dotPrintAnalysis.run(elem.func, s"collect_$sym.dot")

            emitBlock(elem.buf.alloc)

  //          stream.println(s"// The func function - elem.func")
            // emitBlock(elem.func)

            stream.println(s"// The update function - elem.buf.update")
            emitBlock(elem.buf.update)

          case (sym, elem: DeliteReduceElem[_]) =>

            if (isPrimitiveReduce(elem)) {
              // Simple reduces currently use a counter chain
              // This is the way Maxeler's documents describe how to perform
              // the reduction
              stream.println(s"""OffsetExpr ${quote(sym)}_loopLength = stream.makeOffsetAutoLoop(\"${quote(sym)}_loopLength\");""")
              stream.println(s"DFEVar ${quote(sym)}_loopLengthVal = ${quote(sym)}_loopLength.getDFEVar(this, dfeUInt(8));")
              stream.println(s"DFEVar ${quote(sym)}_loopCounter = ${quote(sym)}_chain.addCounter(${quote(sym)}_loopLengthVal, 1);")

              stream.println(s"// The rFunc block")
              stream.println(s"DFEVar ${quote(sym)}_oldAccum = dfeUInt(32).newInstance(this);")
              stream.println(s"DFEVar ${quote(sym)}_zero = constant.var(dfeUInt(32), ${quote(getBlockResult(elem.zero))});")
              emitValDef(elem.rV._2, s"en ? ${quote(getBlockResult(elem.func))} : ${quote(sym)}_zero")
              emitValDef(elem.rV._1, s"${quote(sym)}_oldAccum")
              emitBlock(elem.rFunc)
              stream.println(s"${quote(sym)}_oldAccum <== stream.offset(${quote(getBlockResult(elem.rFunc))}, -${quote(sym)}_loopLength);")

              // Where is the reduced value stored?
              // Current solution: Have DFEVar in BaseKernelLib, assign it to the result from the accumulator
              // This implementation utilizes the FF used to implement the delay (-looplength) to store the
              // accumulation
              baseKernelLibStream.println(s"protected static DFEVar ${quote(sym)};")
              topKernelStream.println(s"""BaseKernelLib.${quote(sym)} = dfeUInt(32).newInstance(this);""")
              stream.println(s"""${quote(sym)} <== ${quote(getBlockResult(elem.rFunc))};""")

  //            val tempPw = new PrintWriter("/home/raghu/work/research/mydir2/hyperdsl/delite/framework/delite-test/testFoo.txt")
  //            withStream(tempPw) {
  //                emitScalarReduceFSM(elem.rFunc)
  //            }
  //            tempPw.close()
  //            val bName = aliasMap(getBlockResult(elem.rFunc))
  //            val fsmName = s"rfunc_${bName}"
  //            stream.println(s"""SMIO $fsmName = addStateMachine(\"$fsmName\", new ScalarReduceFSM_${bName}(this));""");
  //            stream.println(s"""${fsmName}.connectInput(\"sm_d\",  ${quote(aliasMap(elem.rV._1))});""")
  //            stream.println(s"""${fsmName}.connectInput(\"en\", en);""")
            } else {
              aliasMap(elem.rV._1) = sym
              aliasMap(elem.rV._2) = getBlockResult(elem.func)
              aliasMap(getBlockResult(elem.rFunc)) = sym
              aliasMap(getBlockResult(elem.zero)) = sym

              stream.println("// Zero block")
              emitBlock(elem.zero)
              stream.println("// End Zero block")
  //            emitValDef(elem.rV._2, s"${quote(getBlockResult(elem.func))} : ${quote(getBlockResult(elem.zero))}")
  //            emitValDef(elem.rV._1, s"${quote(sym)}_oldAccum")
              stream.println("// rFunc block")
              stream.println(s"// Alias map: $aliasMap")
              emitBlock(elem.rFunc)
              stream.println("// End rFunc block")
  //            sys.error(s"Not handling reduces of non-primitive ${elem.mA} types yet!")
            }

          case _ =>
            throw new Exception("Not handled yet")
        }
        curSym.pop
      }

      symBodyTuple.foreach { t =>
        seenLoops += t._1
      }
      aliasMap = prevAliasMap
    case _ => super.emitFatNode(symList, rhs)
  }

  // Abstract methods in GenericGenDeliteOps defined here
  def quotearg(x: Sym[Any]) = {
  }

  def quotetp(x: Sym[Any]) = {
  }

  def methodCall(name:String, inputs: List[String] = Nil): String = {
    "methodCall"
  }

  def emitMethodCall(name:String, inputs: List[String]): Unit = {
  }

  def emitMethod(name:String, outputType: String, inputs:List[(String,String)])(body: => Unit): Unit = {
    if (Config.debugCodegen) {
      println(s"[codegen] [HwGenDeliteOps] emitMethod ($name, $outputType, $inputs)")
    }
  }

  def createInstance(typeName:String, args: List[String] = Nil): String = {
    "createInstance"
  }

  def fieldAccess(className: String, varName: String): String = {
    "fieldAccess"
  }

  def releaseRef(varName: String): Unit = {
  }

  def emitReturn(rhs: String) = {
  }

  def emitFieldDecl(name: String, tpe: String) = {
  }

  def emitClass(name: String)(body: => Unit) = {
  }

  def emitObject(name: String)(body: => Unit) = {
  }

  def emitValDef(name: String, tpe: String, init: String): Unit = {
  }

  def emitVarDef(name: String, tpe: String, init: String): Unit = {
  }

  def emitAssignment(name: String, tpe: String, rhs: String): Unit = {
  }

  def emitAssignment(lhs: String, rhs: String): Unit = {
  }

  def emitAbstractFatLoopHeader(className: String, actType: String): Unit = {
    if (Config.debugCodegen) {
      println(s"[codegen] Calling emitAbstractFatLoopHeader on classname: $className, actType: $actType")
    }
  }

  def emitAbstractFatLoopFooter(): Unit = {
    if (Config.debugCodegen) {
      println(s"[codegen] Calling emitAbstractFatLoopFooter")
    }
  }

  def castInt32(name: String): String = {
    "castInt32"
  }

  def refNotEq: String = {
    "refNotEq"
  }

  def nullRef: String = {
    "nullRef"
  }

  def arrayType(argType: String): String = {
    "arrayType"
  }

  def arrayApply(arr: String, idx: String): String = {
    "arrayApply"
  }

  def newArray(argType: String, size: String): String = {
    "newArray"
  }

  def hashmapType(argType: String): String = {
    "hashmapType"
  }

  def typeCast(sym: String, to: String): String = {
    "typeCast"
  }

  def withBlock(name: String)(block: => Unit): Unit = {
  }
}

