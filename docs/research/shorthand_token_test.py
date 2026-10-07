import re, tiktoken

PASSAGES = {
"prose": """The meeting has been rescheduled to Thursday afternoon because the client requested additional time to review the proposal. Please make sure that everyone on the team receives the updated calendar invitation and that the conference room is booked for at least two hours.""",
"instruction": """You are a helpful assistant. Summarize the following document in three bullet points. Focus on the main findings and the recommended next steps. Do not include any information that is not present in the document. Respond in plain English without any markdown formatting.""",
"technical": """The function returns a promise that resolves when the database connection is established. If the connection fails after three retries, it throws an error containing the original exception message. Callers should always wrap the invocation in a try-catch block and log the failure before exiting the process.""",
"code": """def fetch_user(user_id: int) -> dict:
    response = requests.get(f"{BASE_URL}/users/{user_id}", timeout=10)
    response.raise_for_status()
    return response.json()""",
}

ABBR = {"because":"b/c","with":"w/","without":"w/o","and":"&","you":"u","are":"r","your":"ur","through":"thru","please":"pls","people":"ppl","before":"b4","to":"2","for":"4","the":"the","should":"shd","would":"wd","could":"cd","information":"info","document":"doc","function":"fn","database":"db","connection":"conn","error":"err","message":"msg","response":"resp","request":"req","additional":"addl","afternoon":"pm","meeting":"mtg","conference":"conf","calendar":"cal","invitation":"invite","summarize":"summ","between":"btw","something":"sth","approximately":"approx","important":"imp","regarding":"re","about":"abt"}
FILLER = {"the","a","an","is","are","be","been","that","this","of","to","please","any","always","very","really","just","make","sure"}
SYM = {"because":"∵","therefore":"∴","and":"&","returns":"→","resolves":"✓","fails":"✗","not":"¬","with":"w/","without":"w/o","at least":"≥","for":"∀"}

def drop_inner_vowels(t):  # keep first/last letter of each word (Teeline-ish)
    def f(m):
        w=m.group(0)
        if len(w)<=3: return w
        return w[0]+re.sub(r'[aeiouAEIOU]','',w[1:-1])+w[-1]
    return re.sub(r'[A-Za-z]+',f,t)
def drop_all_vowels(t):
    return re.sub(r'(?<=[A-Za-z])[aeiou]', '', t)
def abbrev(t):
    return re.sub(r'\b[A-Za-z]+\b', lambda m: ABBR.get(m.group(0).lower(), m.group(0)), t)
def caveman(t):
    return re.sub(r'\s+',' ',re.sub(r'\b[A-Za-z]+\b', lambda m: '' if m.group(0).lower() in FILLER else m.group(0), t)).replace(' ,',',').replace(' .','.')
def symbolic(t):
    s=caveman(abbrev(t))
    for k,v in SYM.items(): s=re.sub(r'\b'+re.escape(k)+r'\b',v,s)
    return s
def nospaces(t): return t.replace(' ','')
def lower(t): return t.lower()

STYLES = {
 "original": lambda t:t,
 "lowercase": lower,
 "caveman (drop filler)": caveman,
 "abbreviations (b/c, w/, info)": abbrev,
 "caveman+abbrev": lambda t: caveman(abbrev(t)),
 "symbolic (∵ → & ¬)": symbolic,
 "Teeline-ish (drop inner vowels)": drop_inner_vowels,
 "drop all non-initial vowels": drop_all_vowels,
 "remove spaces": nospaces,
 "caveman+abbrev, no spaces": lambda t: nospaces(caveman(abbrev(t))),
}

encs = {n: tiktoken.get_encoding(n) for n in ["cl100k_base","o200k_base"]}
for pname, text in PASSAGES.items():
    print(f"\n=== {pname} ===")
    base = {n:len(e.encode(text)) for n,e in encs.items()}
    print(f"{'style':34} {'chars':>6} {'cl100k':>8} {'Δ%':>6} {'o200k':>8} {'Δ%':>6}")
    for sname, fn in STYLES.items():
        s = fn(text)
        row = [f"{sname:34}", f"{len(s):6d}"]
        for n,e in encs.items():
            c=len(e.encode(s)); row += [f"{c:8d}", f"{100*(c-base[n])/base[n]:+5.0f}%"]
        print(" ".join(row))
    print("  sample ->", drop_inner_vowels(text)[:90])
    print("  sample ->", symbolic(text)[:90])

# word-level: how many tokens does each transform cost per word?
print("\n=== per-word examples (o200k) ===")
e=encs["o200k_base"]
for w in ["because","b/c","bcs","phone","fn","fon","information","info","infrmtn","database","db","dtbs","connection","conn","cnnctn","the","please","pls","through","thru","thr"]:
    print(f"{w:12} {len(e.encode(' '+w))} tok  {[e.decode([t]) for t in e.encode(' '+w)]}")
