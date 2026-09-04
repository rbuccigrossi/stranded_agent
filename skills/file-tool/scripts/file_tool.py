#!/usr/bin/env python3
import argparse,shutil,mimetypes
def text(p):
 with open(p,encoding="utf-8",newline="") as f:return f.read()
def line_text(value, lines):
 if value.endswith((chr(10), chr(13))): return value
 for line in lines:
  if line.endswith(chr(13)+chr(10)): return value + chr(13)+chr(10)
  if line.endswith((chr(10), chr(13))): return value + line[-1]
 return value + chr(10)
def has_final_newline(lines):
 return bool(lines) and lines[-1].endswith((chr(10), chr(13)))
def main():
 a=argparse.ArgumentParser(description="Inspect and edit text files");s=a.add_subparsers(dest="op",required=True)
 for n in ("info","view","search","replace","edit-lines","insert-lines","delete-lines","copy"):
  p=s.add_parser(n);p.add_argument("path");p.add_argument("args",nargs="*")
 x=a.parse_args();p=x.path
 if x.op=="info":
  t=text(p);print(f"path: {p}\nsize: {len(t.encode()):,} bytes\nlines: {len(t.splitlines())}\ncharacters: {len(t)}\ntype: {mimetypes.guess_type(p)[0] or 'unknown'}");return
 if x.op=="copy":shutil.copy2(p,x.args[0]);print(f"Copied {p} to {x.args[0]}");return
 q=text(p).splitlines(True)
 if x.op=="view":
  lo=int(x.args[0]) if x.args else 1;hi=int(x.args[1]) if len(x.args)>1 else len(q);print("".join(f"{i}: {v}" for i,v in enumerate(q[lo-1:hi],lo)),end="");return
 if x.op=="search":
  h=[(i,v) for i,v in enumerate(q,1) if x.args[0] in v];print("".join(f"{i}: {v}" for i,v in h),end="");print(f"{len(h)} match(es)");return
 backup="--backup" in x.args
 if backup:shutil.copy2(p,p+".bak")
 if x.op=="replace":t=text(p).replace(x.args[0],x.args[1],int(x.args[2]) if len(x.args)>2 and x.args[2]!="--backup" else -1)
 else:
  i=int(x.args[0])-1;j=int(x.args[1]) if x.op!="insert-lines" else i
  if x.op=="delete-lines":del q[i:j]
  elif x.op=="insert-lines":q[i:i]=[line_text(x.args[1],q)]
  else:q[i:j]=[line_text(x.args[2],q[i:j] or q)] if has_final_newline(q) or j<len(q) else [x.args[2]]
  t="".join(q)
 with open(p,"w",encoding="utf-8",newline="") as f:f.write(t)
 print(f"Updated {p}")
if __name__=="__main__":main()
