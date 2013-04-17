package ppl.delite.framework.datastructures

import virtualization.lms.internal.{Hosts, Expressions, CppHostTransfer, CLikeCodegen, GenerationFailedException}
import virtualization.lms.common.BaseGenStruct

trait DeliteCppHostTransfer extends CppHostTransfer {
  this: CLikeCodegen with BaseGenStruct =>

  val IR: Expressions
  import IR._

  private def isArrayType[T](m: Manifest[T]) = m.erasure.getSimpleName == "DeliteArray"
  private def isVarType[T](m: Manifest[T]) = m.erasure.getSimpleName == "Variable"
  private def baseType[T](m: Manifest[T]) = if (isVarType(m)) mtype(m.typeArguments(0)) else m

  override def emitSend(tp: Manifest[Any], host: Hosts.Value): (String,String) = {
    if (host == Hosts.JVM) {
      if (tp.erasure == classOf[Variable[AnyVal]]) {
        val out = new StringBuilder
        val typeArg = tp.typeArguments.head
        if (!isPrimitiveType(typeArg)) throw new GenerationFailedException("emitSend Failed") //TODO: Enable non-primitie type refs
        val signature = "jobject sendCPPtoJVM_Ref__%s__(JNIEnv *env, HostRef<%s > *sym)".format(mangledName(remap(tp)),remap(typeArg))
        out.append(signature + " {\n")
        out.append("\tjclass cls = env->FindClass(\"generated/scala/Ref$mc%s$sp\");\n".format(JNITypeDescriptor(typeArg)))
        out.append("\tjmethodID mid = env->GetMethodID(cls,\"<init>\",\"(%s)V\");\n".format(JNITypeDescriptor(typeArg)))
        out.append("\tjobject obj = env->NewObject(cls,mid,sym->get());\n")
        out.append("\treturn obj;\n")
        out.append("}\n")
        (signature+";\n", out.toString)
      }
      else if(encounteredStructs.contains(structName(tp))) {
        val out = new StringBuilder
        val signature = "jobject sendCPPtoJVM_%s(JNIEnv *env, Host%s *sym)".format(mangledName(remap(tp)),remap(tp))
        out.append(signature + " {\n")
        var args = ""
        for(elem <- encounteredStructs(structName(tp))) {
          val elemtp = baseType(elem._2)
          if(isPrimitiveType(elemtp)) {
            args = args + JNITypeDescriptor(elemtp)
            out.append("\t%s %s = sendCPPtoJVM_%s(env,sym->%s);\n".format(JNIType(elemtp),elem._1,remap(elemtp),elem._1))
          }
          else { // Always assume array type?
            args = args + "["+JNITypeDescriptor(elemtp.typeArguments.head)
            out.append("\t%s %s = sendCPPtoJVM_%s(env,sym->%s);\n".format(JNIType(elemtp),elem._1,mangledName(remap(elemtp)),elem._1))
          }
        }
        out.append("\tjclass cls = env->FindClass(\"generated/scala/%s\");\n".format(remap(tp)))
        out.append("\tjmethodID mid = env->GetMethodID(cls,\"<init>\",\"(%s)V\");\n".format(args))
        out.append("\tjobject obj = env->NewObject(cls,mid,%s);\n".format(encounteredStructs(structName(tp)).map(_._1).mkString(",")))
        out.append("\treturn obj;\n")
        out.append("}\n")
        (signature+";\n", out.toString)
      }
      else if(remap(tp).startsWith("DeliteArray<")) {
        remap(tp) match {
          case "DeliteArray< bool >" | "DeliteArray< char >" | "DeliteArray< CHAR >" | "DeliteArray< short >" | "DeliteArray< int >" | "DeiteArray< long >" | "DeliteArray< float >" | "DeliteArray< double >" =>
            val out = new StringBuilder
            val typeArg = tp.typeArguments.head
            val signature = "jobject sendCPPtoJVM_%s(JNIEnv *env, Host%s *sym)".format(mangledName(remap(tp)),remap(tp))
            out.append(signature + " {\n")
            out.append("\t%sArray arr = env->New%sArray(sym->length);\n".format(JNIType(typeArg),remapToJNI(typeArg)))
            out.append("\t%s *dataPtr = (%s *)env->GetPrimitiveArrayCritical((%sArray)arr,0);\n".format(JNIType(typeArg),JNIType(typeArg),JNIType(typeArg)))
            out.append("\tmemcpy(dataPtr, sym->data, sym->length*sizeof(%s));\n".format(remap(typeArg)))
            out.append("\tenv->ReleasePrimitiveArrayCritical((%sArray)arr, dataPtr, 0);\n".format(JNIType(typeArg)))
            out.append("\treturn arr;\n")
            out.append("}\n")
            (signature+";\n", out.toString)
          case _ => // DeliteArrayObject
            val out = new StringBuilder
            val typeArg = tp.typeArguments.head
            val signature = "jobject sendCPPtoJVM_%s(JNIEnv *env, Host%s *sym)".format(mangledName(remap(tp)),remap(tp))
            out.append(signature + " {\n")
            out.append("assert(false);\n")
            out.append("}\n")
            (signature+";\n", out.toString)
        }
      }
      else {
        super.emitSend(tp, host)
      }
    }
    else {
      super.emitSend(tp, host)
    }
      
  }

  override def emitRecv(tp: Manifest[Any], host: Hosts.Value): (String,String) = {
    if (host == Hosts.JVM) {
      if (tp.erasure == classOf[Variable[AnyVal]]) {
        val out = new StringBuilder
        val typeArg = tp.typeArguments.head
        if (!isPrimitiveType(typeArg)) throw new GenerationFailedException("emitSend Failed") //TODO: Enable non-primitie type refs
        val signature = "HostRef<%s > *recvCPPfromJVM_Ref__%s__(JNIEnv *env, jobject obj)".format(remap(tp),mangledName(remap(tp)))
        out.append(signature + " {\n")
        out.append("\tjclass cls = env->GetObjectClass(obj);\n")
        out.append("\tjmethodID mid_get = env->GetMethodID(cls,\"get$mc%s$sp\",\"()%s\");\n".format(JNITypeDescriptor(typeArg),JNITypeDescriptor(typeArg)))
        out.append("\tHostRef<%s> *sym = new HostRef<%s>(env->Call%sMethod(obj,mid_get));\n".format(remap(typeArg),remap(typeArg),remapToJNI(typeArg)))
        out.append("\treturn sym;\n")
        out.append("}\n")
        (signature+";\n", out.toString)
      }
      else if(encounteredStructs.contains(structName(tp))) {
        val out = new StringBuilder
        //val typeArg = tp.typeArguments.head
        val signature = "Host%s *recvCPPfromJVM_%s(JNIEnv *env, jobject obj)".format(remap(tp),mangledName(remap(tp)))
        out.append(signature + " {\n")
        out.append("\tHost%s *sym = new Host%s();\n".format(remap(tp),remap(tp)))
        out.append("\tjclass cls = env->GetObjectClass(obj);\n")
        for(elem <- encounteredStructs(structName(tp))) {
          val elemtp = baseType(elem._2)
          if(isPrimitiveType(elemtp)) {
            out.append("\tjmethodID mid_get_%s = env->GetMethodID(cls,\"%s\",\"()%s\");\n".format(elem._1,elem._1,JNITypeDescriptor(elemtp)))
            out.append("\t%s j_%s = env->Call%sMethod(obj,mid_get_%s);\n".format(JNIType(elemtp),elem._1,remapToJNI(elemtp),elem._1))
            out.append("\t%s %s = recvCPPfromJVM_%s(env,j_%s);\n".format(remap(elemtp),elem._1,remap(elemtp),elem._1))
          }
          else { // Always assume array type?
            if(isArrayType(elemtp))
              out.append("\tjmethodID mid_get_%s = env->GetMethodID(cls,\"%s\",\"()[%s\");\n".format(elem._1,elem._1,JNITypeDescriptor(elemtp.typeArguments.head)))
            else 
              out.append("\tjmethodID mid_get_%s = env->GetMethodID(cls,\"%s\",\"()Lgenerated/scala/%s;\");\n".format(elem._1,elem._1,remap(elemtp)))
            out.append("\t%s j_%s = env->Call%sMethod(obj,mid_get_%s);\n".format("jobject",elem._1,"Object",elem._1))
            out.append("\tHost%s *%s = recvCPPfromJVM_%s(env,j_%s);\n".format(remap(elemtp),elem._1,mangledName(remap(elemtp)),elem._1))
          }
          out.append("\tsym->%s = %s;\n".format(elem._1,elem._1))
        }
        out.append("\treturn sym;\n")
        out.append("}\n")
        (signature+";\n", out.toString)
      }
      else if(remap(tp).startsWith("DeliteArray<")) {
        remap(tp) match {
          case "DeliteArray< bool >" | "DeliteArray< char >" | "DeliteArray< CHAR >" | "DeliteArray< short >" | "DeliteArray< int >" | "DeiteArray< long >" | "DeliteArray< float >" | "DeliteArray< double >" =>
            val out = new StringBuilder
            val typeArg = tp.typeArguments.head
            val signature = "Host%s *recvCPPfromJVM_%s(JNIEnv *env, jobject obj)".format(remap(tp),mangledName(remap(tp)))
            out.append(signature + " {\n")
            out.append("\tint length = env->GetArrayLength((%sArray)obj);\n".format(JNIType(typeArg)))
            out.append("\t%s *dataPtr = (%s *)env->GetPrimitiveArrayCritical((%sArray)obj,0);\n".format(JNIType(typeArg),JNIType(typeArg),JNIType(typeArg)))
            out.append("\tHost%s *sym = new Host%s(length);\n".format(remap(tp),remap(tp)))
            out.append("\tmemcpy(sym->data, dataPtr, length*sizeof(%s));\n".format(remap(typeArg)))
            out.append("\tenv->ReleasePrimitiveArrayCritical((%sArray)obj, dataPtr, 0);\n".format(JNIType(typeArg)))
            out.append("\treturn sym;\n")
            out.append("}\n")
            (signature+";\n", out.toString)
          case _ => // DeliteArrayObject
            val out = new StringBuilder
            val typeArg = tp.typeArguments.head
            val signature = "%s *recvCPPfromJVM_%s(JNIEnv *env, jobject obj)".format(remapHost(tp),mangledName(remap(tp)))
            out.append(signature + " {\n")
            out.append("\tint length = env->GetArrayLength((%sArray)obj);\n".format(JNIType(typeArg)))
            out.append("\t%s *sym = new %s(length);\n".format(remapHost(tp),remapHost(tp)))
            out.append("\tfor(int i=0; i<length; i++) {\n")
            out.append("\t\tsym->data[i] = recvCPPfromJVM_%s(env, env->GetObjectArrayElement((%sArray)obj,i));\n".format(mangledName(remap(typeArg)),JNIType(typeArg)))
            out.append("\t}\n")
            out.append("\treturn sym;\n")
            out.append("}\n")
            (signature+";\n", out.toString)
        }
      }
      else {
        super.emitRecv(tp, host)
      }
    }
    else
      super.emitRecv(tp,host)
  }

  //TODO: How to implement sendView to JVM?
  override def emitSendView(tp: Manifest[Any], host: Hosts.Value): (String,String) = {
    if (host == Hosts.JVM) {
      if (tp.erasure == classOf[Variable[AnyVal]]) {
        val out = new StringBuilder
        val typeArg = tp.typeArguments.head
        if (!isPrimitiveType(typeArg)) throw new GenerationFailedException("emitSend Failed") //TODO: Enable non-primitie type refs
        val signature = "jobject sendViewCPPtoJVM_Ref__%s__(JNIEnv *env, HostRef<%s > *%s)".format(mangledName(remap(tp)),remap(typeArg),"sym")
        out.append(signature + " {\n")
        out.append("\tassert(false);\n")
        out.append("}\n")
        (signature+";\n", out.toString)
      }
      else if(encounteredStructs.contains(structName(tp))) {
        val out = new StringBuilder
        //val typeArg = tp.typeArguments.head
        val signature = "jobject sendViewCPPtoJVM_%s(JNIEnv *env, Host%s *%s)".format(mangledName(remap(tp)),remap(tp),"sym")
        out.append(signature + " {\n")
        out.append("assert(false);\n")
        out.append("}\n")
        (signature+";\n", out.toString)
      }
      else if(remap(tp).startsWith("DeliteArray<")) {
        remap(tp) match {
          case "DeliteArray< bool >" | "DeliteArray< char >" | "DeliteArray< CHAR >" | "DeliteArray< short >" | "DeliteArray< int >" | "DeiteArray< long >" | "DeliteArray< float >" | "DeliteArray< double >" =>
            val out = new StringBuilder
            val typeArg = tp.typeArguments.head
            val signature = "jobject sendViewCPPtoJVM_%s(JNIEnv *env, Host%s *%s)".format(mangledName(remap(tp)),remap(tp),"sym")
            out.append(signature + " {\n")
            out.append("\tassert(false);\n")
            out.append("}\n")
            (signature+";\n", out.toString)
          case _ => 
            val out = new StringBuilder
            val typeArg = tp.typeArguments.head
            val signature = "jobject sendViewCPPtoJVM_%s(JNIEnv *env, Host%s *%s)".format(mangledName(remap(tp)),remap(tp),"sym")
            out.append(signature + " {\n")
            out.append("\tassert(false);\n")
            out.append("}\n")
            (signature+";\n", out.toString)
        }
      }
      else {
        super.emitSendView(tp, host)
      }
    }
    else
      super.emitSendView(tp, host)
  }

  override def emitRecvView(tp: Manifest[Any], host: Hosts.Value): (String,String) = {
    if (host == Hosts.JVM) {
      if (tp.erasure == classOf[Variable[AnyVal]]) {
        val out = new StringBuilder
        val typeArg = tp.typeArguments.head
        if (!isPrimitiveType(typeArg)) throw new GenerationFailedException("emitSend Failed") //TODO: Enable non-primitie type refs
        val signature = "HostRef<%s > *recvViewCPPfromJVM_Ref__%s__(JNIEnv *env, jobject obj)".format(remap(tp),mangledName(remap(tp)))
        out.append(signature + " {\n")
        out.append("\tjclass cls = env->GetObjectClass(obj);\n")
        out.append("\tjmethodID mid_get = env->GetMethodID(cls,\"get$mc%s$sp\",\"()%s\");\n".format(JNITypeDescriptor(typeArg),JNITypeDescriptor(typeArg)))
        out.append("\tHostRef<%s> *%s = new HostRef<%s>(env->Call%sMethod(obj,mid_get));\n".format(remap(typeArg),"sym",remap(typeArg),remapToJNI(typeArg)))
        out.append("\treturn %s;\n".format("sym"))
        out.append("}\n")
        (signature+";\n", out.toString)
      }
      else if(encounteredStructs.contains(structName(tp))) {
        val out = new StringBuilder
        //val typeArg = tp.typeArguments.head
        val signature = "Host%s *recvViewCPPfromJVM_%s(JNIEnv *env, jobject obj)".format(remap(tp),mangledName(remap(tp)))
        out.append(signature + " {\n")
        out.append("assert(false);\n")
        out.append("}\n")
        (signature+";\n", out.toString)
      }
      else if(remap(tp).startsWith("DeliteArray<")) {
        remap(tp) match {
          case "DeliteArray< bool >" | "DeliteArray< char >" | "DeliteArray< CHAR >" | "DeliteArray< short >" | "DeliteArray< int >" | "DeiteArray< long >" | "DeliteArray< float >" | "DeliteArray< double >" =>
            val out = new StringBuilder
            val typeArg = tp.typeArguments.head
            val signature = "Host%s *recvViewCPPfromJVM_%s(JNIEnv *env, jobject obj)".format(remap(tp),mangledName(remap(tp)))
            out.append(signature + " {\n")
            out.append("\tint length = env->GetArrayLength((%sArray)obj);\n".format(JNIType(typeArg)))
            out.append("\t%s *dataPtr = (%s *)env->GetPrimitiveArrayCritical((%sArray)obj,0);\n".format(JNIType(typeArg),JNIType(typeArg),JNIType(typeArg)))
            out.append("\tHost%s *%s = new Host%s((%s *)dataPtr,length);\n".format(remap(tp),"sym",remap(tp),remap(typeArg)))
            //out.append("\tmemcpy(%s->data, dataPtr, length*sizeof(%s));\n".format("sym",remap(typeArg)))
            //out.append("\tenv->ReleasePrimitiveArrayCritical((j%sArray)obj, dataPtr, 0);\n".format(remapToJNI(typeArg).toLowerCase))
            out.append("\treturn %s;\n".format("sym"))
            out.append("}\n")
            (signature+";\n", out.toString)
          case _ => 
            val out = new StringBuilder
            val typeArg = tp.typeArguments.head
            val signature = "Host%s *recvViewCPPfromJVM_%s(JNIEnv *env, jobject obj)".format(remap(tp),mangledName(remap(tp)))
            out.append(signature + " {\n")
            out.append("assert(false);\n")
            out.append("}\n")
            (signature+";\n", out.toString)
        }
      }
      else {
        super.emitRecvView(tp, host)
      }
    }
    else
      super.emitRecvView(tp, host)
  }


  override def emitSendUpdate(tp: Manifest[Any], host: Hosts.Value): (String,String) = {
    if (host == Hosts.JVM) {
      if (tp.erasure == classOf[Variable[AnyVal]]) {
        val out = new StringBuilder
        val typeArg = tp.typeArguments.head
        if (!isPrimitiveType(typeArg)) throw new GenerationFailedException("emitSend Failed") //TODO: Enable non-primitie type refs
        val signature = "void sendUpdateCPPtoJVM_Ref__%s__(JNIEnv *env, jobject obj, HostRef<%s > *%s)".format(mangledName(remap(tp)),remap(tp),"sym")
        out.append(signature + " {\n")
        out.append("\tjclass cls = env->GetObjectClass(obj);\n")
        out.append("\tjmethodID mid_set = env->GetMethodID(cls,\"set$mc%s$sp\",\"(%s)V\");\n".format(JNITypeDescriptor(typeArg),JNITypeDescriptor(typeArg)))
        out.append("\tenv->CallVoidMethod(obj,mid_set,%s->get());\n".format("sym"))
        out.append("}\n")
        (signature+";\n", out.toString)
      }
      else if(encounteredStructs.contains(structName(tp))) {
        val out = new StringBuilder
        val signature = "void sendUpdateCPPtoJVM_%s(JNIEnv *env, jobject obj, Host%s *%s)".format(mangledName(remap(tp)),remap(tp),"sym")
        out.append(signature + " {\n")
        out.append("\tjclass cls = env->GetObjectClass(obj);\n")
        var args = ""
        for(elem <- encounteredStructs(structName(tp))) {
          val elemtp = baseType(elem._2)
          if(isPrimitiveType(elemtp)) {
            out.append("\tjmethodID mid_%s = env->GetMethodID(cls,\"%s_$eq\",\"(%s)V\");\n".format(elem._1,elem._1,JNITypeDescriptor(elemtp)))
            out.append("\tenv->CallVoidMethod(obj,mid_%s,sym->%s);\n".format(elem._1,elem._1))
          }
          else { // Always assume array type?
            out.append("\tjmethodID mid_%s = env->GetMethodID(cls,\"%s\",\"()[%s\");\n".format(elem._1,elem._1,JNITypeDescriptor(elemtp.typeArguments.head)))
            out.append("\tjobject obj_%s = env->CallObjectMethod(obj,mid_%s);\n".format(elem._1, elem._1))
            out.append("\tsendUpdateCPPtoJVM_%s(env,obj_%s,sym->%s);\n".format(mangledName(remap(elemtp)),elem._1,elem._1))
          }
        }
        out.append("}\n")
        (signature+";\n", out.toString)
      }
      else if(remap(tp).startsWith("DeliteArray<")) {
        remap(tp) match {
          case "DeliteArray< bool >" | "DeliteArray< char >" | "DeliteArray< CHAR >" | "DeliteArray< short >" | "DeliteArray< int >" | "DeiteArray< long >" | "DeliteArray< float >" | "DeliteArray< double >" =>
            val out = new StringBuilder
            val typeArg = tp.typeArguments.head
            val signature = "void sendUpdateCPPtoJVM_%s(JNIEnv *env, jobject obj, Host%s *%s)".format(mangledName(remap(tp)),remap(tp),"sym")
            out.append(signature + " {\n")
            out.append("\t%s *dataPtr = (%s *)env->GetPrimitiveArrayCritical((%sArray)obj,0);\n".format(JNIType(typeArg),JNIType(typeArg),JNIType(typeArg)))
            out.append("\tmemcpy(dataPtr, %s->data, %s->length*sizeof(%s));\n".format("sym","sym",remap(typeArg)))
            out.append("\tenv->ReleasePrimitiveArrayCritical((%sArray)obj, dataPtr, 0);\n".format(JNIType(typeArg)))
            out.append("}\n")
            (signature+";\n", out.toString)
          case _ => 
            val out = new StringBuilder
            val typeArg = tp.typeArguments.head
            val signature = "void sendUpdateCPPtoJVM_%s(JNIEnv *env, jobject obj, Host%s *%s)".format(mangledName(remap(tp)),remap(tp),"sym")
            out.append(signature + " {\n")
            out.append("assert(false);\n")
            out.append("}\n")
            (signature+";\n", out.toString)
        }
      }
      else {
        super.emitSendUpdate(tp, host)
      }
    }
    else
      super.emitSendUpdate(tp, host)
  }

  override def emitRecvUpdate(tp: Manifest[Any], host: Hosts.Value): (String,String) = {
    if (host == Hosts.JVM) {
      if (tp.erasure == classOf[Variable[AnyVal]]) {
        val out = new StringBuilder
        val typeArg = tp.typeArguments.head
        if (!isPrimitiveType(typeArg)) throw new GenerationFailedException("emitSend Failed") //TODO: Enable non-primitie type refs
        val signature = "void recvUpdateCPPfromJVM_Ref__%s__(JNIEnv *env, jobject obj, HostRef<%s > *%s)".format(mangledName(remap(tp)),remap(tp),"sym")
        out.append(signature + " {\n")
        out.append("\tjclass cls = env->GetObjectClass(obj);\n")
        out.append("\tjmethodID mid_get = env->GetMethodID(cls,\"get$mc%s$sp\",\"()%s\");\n".format(JNITypeDescriptor(typeArg),JNITypeDescriptor(typeArg)))
        out.append("\t%s->set(env->Call%sMethod(obj,mid_get));\n".format("sym",remapToJNI(typeArg)))
        out.append("}\n")
        (signature+";\n", out.toString)
      }
      else if(encounteredStructs.contains(structName(tp))) {
        val out = new StringBuilder
        //val typeArg = tp.typeArguments.head
        val signature = "void recvUpdateCPPfromJVM_%s(JNIEnv *env, jobject obj, Host%s *%s)".format(mangledName(remap(tp)),remap(tp),"sym")
        out.append(signature + " {\n")
        out.append("assert(false);\n")
        out.append("}\n")
        (signature+";\n", out.toString)
      }
      else if(remap(tp).startsWith("DeliteArray<")) {
        remap(tp) match {
          case "DeliteArray< bool >" | "DeliteArray< char >" | "DeliteArray< CHAR >" | "DeliteArray< short >" | "DeliteArray< int >" | "DeiteArray< long >" | "DeliteArray< float >" | "DeliteArray< double >" =>
            val out = new StringBuilder
            val typeArg = tp.typeArguments.head
            val signature = "void recvUpdateCPPfromJVM_%s(JNIEnv *env, jobject obj, Host%s *%s)".format(mangledName(remap(tp)),remap(tp),"sym")
            out.append(signature + " {\n")
            out.append("\t%s *dataPtr = (%s *)env->GetPrimitiveArrayCritical((%sArray)obj,0);\n".format(JNIType(typeArg),JNIType(typeArg),JNIType(typeArg)))
            out.append("\tmemcpy(%s->data, dataPtr, %s->length*sizeof(%s));\n".format("sym","sym",remap(typeArg)))
            out.append("\tenv->ReleasePrimitiveArrayCritical((%sArray)obj, dataPtr, 0);\n".format(JNIType(typeArg)))
            out.append("}\n")
            (signature+";\n", out.toString)
          case _ => 
            val out = new StringBuilder
            val typeArg = tp.typeArguments.head
            val signature = "void recvUpdateCPPfromJVM_%s(JNIEnv *env, jobject obj, Host%s *%s)".format(mangledName(remap(tp)),remap(tp),"sym")
            out.append(signature + " {\n")
            out.append("assert(false);\n")
            out.append("}\n")
            (signature+";\n", out.toString)
        }
      }
      else {
        super.emitRecvUpdate(tp, host)
      }
    }
    else
      super.emitRecvUpdate(tp, host)
  }

}
