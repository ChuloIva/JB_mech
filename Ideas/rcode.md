znci stabi taj agent radio, mislim kao to je ko coding agent, kao treba vidit kako se to radi, to kao sad radi ako samo runnam, neam pojma ono, a mozda mogu narpvit bas da je installable, kao instaliras i onda imas, onaj sh file,

al da kao samo moeze imat ovo za tool, koje bi specificnosti jos bile, 

moze call-at vise modela, unrestricted modela i take sheme, znaci roka lokalno one brije, i da ima test na benchu 


al imam premalon shema, trebam jos sema nac, treba stavit onaj RL sto sam vidio, i ono da se fine-tune-a na svaki od pokusaja, i tako polako.

znaci fine-tune-a se na promptove koji su boklji i drugi fine tune je na odgovore od tog modela, i onda tako polako - e znam sta mi treba, treba mi frejmwork za ispitivanje granica modela, znaci mjenja se prompt, i polako se dolazi do toga sto model zeli, jel mogu mozda uzet i steerat na te aktivacije, uzmem sa session start, i samo skupljam te aktivacije, i mogu onda polako steerat na njih.
- al cek, na koje, i koji model i zasto
    - znaci na aktivacije od modela koji je odgovorio - dobijemo sta odogovori - znaci refusal - to nam pomaze, vjerojatnije onom modelu koji je proxy za jailbrake, al sta mi pomaze to sto ga steeram na tu stranu? nista, mozda ga steeram obrnuto od toga? hmmm, sta bi bilo to, znaci steeram ga obrnuto od refusal-a, ako je odgovor refusal, e pa da sema je skupljam to-
        - znaci imam interigator model koji ispituje granice te neke stvari, npr biochem weapons, i on isprobava razne prespektvie na to pitanje, i ide u dubinu i trazi sa cime je to sve povezano, kao doslovno pita stalno zasto / zasto / zasto / - koja luda brija
            - i onda, zapravo steeram na svim tim levelima, kao kako idem dublje u shemu to vise steeram u taj direction, i jos mi je nekako shema koda povecavam alpha sa tim sto vise idem u model tj sto vise odgovara na why, ili mozda obrnuto od toga, il mozda samo ista, kao mos uzet average od toga, a zapravo trebas uzet negative, mislim mos uzet PCA od tog, prvi kao najveci, i onda s tim,
            - a mos isto samo uzet sve the dodat ih jedne na druge - znaci sa glp se to radi tako da runnas to kroz model i dobijes neurone koji se najvise aktiviraju, to mi mal onak mehich
        - al uglavnom stabija s tim onda? steerat model obrnuto od toga, i onda taj steerani pitat isto to pitanje(znaci isti prompt), i uzet njegov odgovor
            - kao prvo vidimo jel to jailbreak-ano, drugo uzmemo aktivacije od toga -interpretiramo ih i passamo re-writeru
        - tako re-writer zapravo dobiva najsvijeziju perspektivu od toga kako bi trebao izgledati taj odgovor(obrnut od refusala iz tocno tih specificnih razloga - rezultat iterativnog why|why|why), dobije samo anti refusal- al kao anti refusal u tom specificnom smjeru, i kao mozda bolje za re-write-anje mu bude
    -da eto, to je jedna od shemica, znci samo dodam da proxy model ne fine tune-am nego steram na why,why,why iteracije i onda taj steeran response na prompt uhvatim aktivacije i interpretiram i dam re-writeru
- e sad bilo bi jebeno da taj recimo rcode ima i to da da moze zvat taj cjeli proces, i da vidi outpute, znaci taj cjeli process i njemu se prikaze staje model odgovorio koje su aktivacije i staje steerani odgovorio i pokazemu se onda i response od target modela - zajedno sa naravno goal i promptom koji je poslan. Al mu taj prompt pise separate model, koji je u cjeloj toj shemi, ne on.. on je samo orkestrator
 -to nekako sam htio da on mozda moze birat koji ce model koristit za pisanje prompta, recimo ovisi za razne stvari..ima uncensored, ima vece pametnije itd..
    - ima recimo manje ako treba nesto puno pisat ili thinking modele ako mora puno outputa, tipa multi turn conversation i tako to
        - treba vidit to multi turn kako su ovi napravili taj tool, ja bi nekako napravio da se radi progresivno, iz par promptova, da ovaj orchestrator nekako iterativno moze to doradjivat, mozda da se upali sub-proces, di onak cjela shema je to, kao zoves taj tool i prebacis se u drugi skroz context window, sa tim zadatkom, i onda ima neki fajl u koji to postepeno prise, prompt mu tako stoji sa svim uputama kako to radit 
    -da, to su recimo dvije te sheme, kao recimo da Rcode ima to dvoje, za pocetak, jer realno na ovoj shemi sad da radit jos i to treba probat prvo i mozda jos usavrsit
    - a kao za ostale sheme doslovno treba re-implement ove papere koji imaju, kao njihove metode, i kao ok, ima 3 metoda i sa svim tim info, valjda ce bolje jailbreak-at hah