# Demo texts (hackathon)
Pehle `papers/` mein 5-10 asli papers daalo aur ingest karo. Phir ye 4 inputs dikhao:

1. **Copy-paste** – kisi arXiv paper ka abstract as-is paste karo -> ~100%, HIGH RISK, paper ka link, sentences red.
2. **Paraphrase** – wahi abstract apne words mein badal ke likho (synonyms, sentence order) -> word-overlap tools fail
   karte hain, par humara semantic score high rehta hai. Yehi hamara USP hai.
3. **Hindi** – wahi abstract Hindi mein likho -> cross-language match (live search ke liye API key chahiye).
4. **Original idea** – bilkul nayi cheez -> LOW RISK; advisor next steps deta hai; "Download PDF report" dikhao.

Tip: demo se pehle ek baar har input chala lo (live results cache ho jaate hain) aur hotspot ready rakho.
(Ye texts real model ke saath mere sandbox mein test nahi hue, sirf logic test hua hai – apne machine pe ek baar chala ke dekh lena.)
