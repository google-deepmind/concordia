# Deep Time

A planetary sandbox built on Concordia, built around the ReactiveX pub/sub
components in `concordia/components/agent/pubsub.py`. The planet is a web of
named topics: the climate publishes temperature, the biosphere publishes a
census, and every entity perceives the world only through `Subscriber`
components bound to those topics.

## Quickstart

Offline, with the deterministic mock model (no network, about 0.3 s):

```bash
python -m examples.deep_time.run --scenario=daisyworld --mode=mock
```

Live on Together AI (the key is read from `TOGETHER_API_KEY` or
`TOGETHER_AI_API_KEY` and passed explicitly to the wrapper):

```bash
python -m examples.deep_time.run --scenario=daisyworld --mode=together \
    --ticks=10 --max_calls=50
```

Each run writes a folder (`--out_dir`, default `/tmp/deep_time_runs/...`):

| File             | Contents                                                |
| ---------------- | ------------------------------------------------------- |
| `report.html`    | self-contained report: charts, cognition meters,        |
:                  : chronicle, briefings, events, topics, data table        :
| `summary.json`   | outcome, final state, messages per topic, LLM calls and |
:                  : tokens, acts per entity, instruments and epistemic gap  :
| `timeseries.csv` | one row per tick: luminosity, temperature, covers, Ω,   |
:                  : indices                                                 :
| `topics.mmd`     | Mermaid graph of publishers → topics → subscribers      |
| `log.html`       | Concordia's standard `SimulationLog` HTML               |

## Play from a phone

A small web server lets a person play a role, or watch the mock play every role,
from a phone browser:

```bash
python -m examples.deep_time.web.server --port 8890 \
    --demo_dir /tmp/deep_time_demos --make_demos
```

It listens on the loopback interface only and has no authentication. Reach it
from your own devices through a private tunnel, for example `tailscale serve
--bg --https=10004 http://127.0.0.1:8890`, and never expose it to the open
internet.

-   **Play.** Pick any role a model plays at the start: the Steward, a species,
    the grazers, a people, Mission Control or the colonists. Your entity acts
    through Concordia's `HumanActComponent`; the server is its `HumanInput`
    reader, and the mock plays everyone else. Each turn the page shows the
    request's context, the whole prompt a model playing your role would read:
    late, noisy instruments, the Advisor's briefing (the Steward's alone), your
    storehouse and your people's plan, which still deliberates in the
    background, or the colony's relay years out of date. The number strip shows
    what your instruments last delivered and how old it is. The true world is
    revealed when the game ends. Playing a people, you answer treaty offers
    yourself: an offer's terms appear in your sheet, and accepting or declining
    it is on your menu. A people's sheet (and its card) also says what the age
    asks of all the peoples (the iron or energy to make by the last turn, the
    pace that needs and where last turn's pace leads), what held its own making
    back last generation (e.g. iron by charcoal, charcoal by wood: more hands to
    cut wood), and its numbers against the floor under which a people breaks and
    ends the age. The move that answers its most urgent crisis, the one that
    strengthens its weakest link and the tech that most helps its goal are
    tagged on the move sheet (and close the question a model playing it reads);
    while research is short of that tech, the sheet says how far off it is.
-   **Omens.** In the Great Acceleration the Steward may speak: its act is
    Concordia's free action, a move as written or an omen, a sign every
    people sees ("a comet") and up to 25 words it is taken to say, for 2 Ω
    every other turn (the `omen` intervention). The peoples never hear the
    words but through a relay: the Cedar league's printers, who own the
    presses, retell each omen a turn later on the Gazette's front page as
    they see it (`components/relay_mind.py`, published on `media.omens`).
    Its stance sways every people's consent for coal by the printers'
    credibility, which the other peoples judge against the sky, as they
    judge any broadsheet. The page opens the move sheet on the composer.
-   **Characters.** The Great Acceleration's peoples are written as people
    (`civilization.characters`: who they are, what they want and fear, old
    grievances, beliefs, how they speak) and the world's conventions are
    stated once (`civilization.conventions`: what a sign is, who prints the
    Gazette); each is a Concordia `Constant` in the people's prompt and its
    plan. No sheet says how to play. After each retelling every people gives
    a one-sentence take in its own voice (`components/take_mind.py`, on
    `civ.<key>.takes`), shown on the peoples' card and in the story; a take
    decides nothing, the people still acts through its moves.
-   **Crises.** Trouble the Steward can see coming (the climate leaving its
    band, strain building, hunger, unrest, dwindling life, a goal's last turns)
    leads the page with a countdown and the moves that answer it. Moves you
    cannot pay for yet are shown under Saving for.
-   **Plan ahead.** The Steward's temperature chart carries its forecast (its
    late readings reckoned forward with the moves it made since). The When row
    above the moves (+1 to +5) plans a move for a later turn instead of sending
    it: up to three plans, one a turn, each sent by the server on its turn (or,
    if it cannot be made then, you are asked and told why). The forecast's
    course counts the plans, so the chart shows where they will leave the
    planet. After the game, the recap's turning points name the steepest swings
    of the true temperature, each with the move that came before it and what you
    read, what your forecast said and what was true when you made it.
-   **Co-op.** Light Delay and Herds are made for two phones: their cards say
    co-op, and Play together seats home and the colony, or the climate and the
    herds.
-   **Play with a friend.** Pick "with a friend as …": each person gets a seat
    (`/?seat=<role>`), the page shows the link for the other phone, each phone
    sees only its own role's prompt and moves, and the world waits for both (the
    waiting phone says whom it waits for). Each player has three minutes a turn;
    out of time, the role makes its most neutral move (waits, keeps its labor,
    holds steady). Two peoples can offer and sign treaties with each other this
    way.
-   **Saved games.** A played game is saved after every turn (Concordia's
    standard checkpoint, beside the telemetry); if the server restarts, the
    start screen offers ▶ Play on, which resumes it from that turn. Three times
    a game, the moves card's ↶ Back to turn N takes the last turn back to play
    it again. A decided game played with the mock's minds can be watched again
    exactly, from its recap or the start screen.
-   **Watch.** The mock plays every role, one step every 0.8 s; the page shows
    the true world beside what the Steward sees.
-   **Reports.** `/demo/` lists the run folders in `--demo_dir` (`--make_demos`
    writes one MOCK run per scenario first).
-   **Live minds.** With `--live_minds together,ollama` and a budget
    (`--live_budget_usd 12`), the new-game card offers live minds besides the
    mock: language models then play every role no person plays and voice the
    Advisor, Chronicler and presses. Together needs `TOGETHER_API_KEY` in the
    server's environment; Ollama runs small local models (`llama3.2:3b`). The
    title shows the game's spend so far and the recap its calls, tokens and
    cost; each live game's spend goes to `spend.json` beside the telemetry, and
    live minds are no longer offered once it reaches the budget. Without a key,
    or with Ollama down, a live game falls back to the mock and says so. The
    minds buttons show what each kind has cost a game, how long a turn took and
    how often the game was won, in the picked world where it has been played,
    and a table compares them there (from the server's ledger, or the
    programme's measurements in `data/minds.yaml`).
-   **Crises and side objectives.** Trouble the role can see coming in its own
    readings leads the page with a countdown and the moves that answer it
    (`crises.py`). Most scenarios also set an optional side objective (keep the
    Methanogen mats, the woods, the wolves…) with a window and a reward in Ω,
    judged on the true world; it is listed on the goal card, read on the Ω
    meter, raised as a crisis in its last turns and tallied in the recap
    (`side:` in `scenarios.yaml`).
-   **The map shows what you can know.** In play the life layer is your census's
    picture, as late as it ("as your census saw it on turn 6 · 2 turns ago"),
    and an aimed sowing is outlined where it took root once your census has seen
    it; fire danger and the census's band temperatures show the same way.
-   **The map explains.** Each crisis shows where its role can place it (a
    capital, a herd, a latitude band, the map's glowing edge), and the turn a
    move acts the map plays it: clouds drifting, sprouts where you sowed, rain,
    embers, iron in the seas, seeds flying to the vault, a people's towns at
    work (`data/motion.yaml`).
-   **Stars.** Every scenario has one to three stars: the goal, and two harder
    ways to reach it (break the ice sooner, make more energy with cleaner air,
    keep more colonists…). The goal card lists them from the start; the recap
    shows the ones earned (`stars:` in `scenarios.yaml`, calibrated on the
    skilled bot, which earns three in at least one game in every scenario at 16
    seeds; `play_audit` counts them).
-   **The campaign.** The start screen lists the scenarios as chapters through
    deep time, from Daisyworld three billion years ago to New Ararat in 2150; a
    star in a chapter opens the next. **Today's challenge** is one chapter's
    scenario on a world seeded by the date (the same for everyone that day). The
    server keeps the best stars of played games in `progress.json` beside the
    telemetry (`campaign.py`, `data/campaign.yaml`). A decided game's recap
    names its chapter and best stars, and offers the next chapter as a button
    once a star has opened it.
-   **The story so far.** A card gathers, newest first, what your role has been
    told: the Advisor's briefings (the Steward's), the Gazette's broadsheets,
    the news you heard and, once the game is decided, the Chronicler's chapters.
    Tap an entry to read it whole.

## Flags

| Flag             | Default                           | Meaning               |
| ---------------- | --------------------------------- | --------------------- |
| `--scenario`     | `daisyworld`                      | scenario to run       |
| `--mode`         | `mock`                            | `mock`, `together` or |
:                  :                                   : `ollama`              :
| `--ticks`        | scenario length                   | maximum ticks         |
| `--out_dir`      | timestamped temp folder           | run folder            |
| `--fast_model`   | `deepseek-ai/DeepSeek-V4.1-Flash` | per-tick acts         |
| `--slow_model`   | `openai/gpt-oss-120b`             | background            |
:                  :                                   : deliberation          :
| `--max_calls`    | 300                               | call cap per live     |
:                  :                                   : model (budget         :
:                  :                                   : governor)             :
| `--seed`         | 0                                 | seed of the planet's  |
:                  :                                   : random processes      :
:                  :                                   : (eruptions)           :
| `--mock_variant` | `default`                         | mock policies:        |
:                  :                                   : `default` (the        :
:                  :                                   : skilled bot),         :
:                  :                                   : `passive` (the main   :
:                  :                                   : roles never act),     :
:                  :                                   : `random` (they act at :
:                  :                                   : random), `no_advice`  :
:                  :                                   : (the Steward ignores  :
:                  :                                   : its Advisor)          :
| `--time_mode`    | `sim`                             | `sim`: slow minds     |
:                  :                                   : land a fixed number   :
:                  :                                   : of ticks after their  :
:                  :                                   : trigger               :
:                  :                                   : (reproducible);       :
:                  :                                   : `async`\: they run on :
:                  :                                   : threads and land      :
:                  :                                   : whenever the model    :
:                  :                                   : answers               :

## Balance sweeps

```bash
python -m examples.deep_time.sweep --scenario=daisyworld --seeds=8 \
    --variants=default,passive,no_advice --csv=/tmp/sweep.csv
```

Prints win rate, ticks survived, Ω left, peak temperature, biodiversity and
eruptions per variant. Use it before and after any balance change.

## Play audit

```bash
python -m examples.deep_time.play_audit --scenarios=snowball --seeds=16
```

Plays a scenario's main role with three bots (skilled, random, passive) and
prints win rates, the skill gaps between them, how often the skilled bot acts
rather than holds, how many moves with a measured push each turn offers (now and
counting the moves it is saving for), the skilled bot's stars and each bot's
share of side objectives met. The phone server writes every played game to
`<demo_dir>-telemetry/` (outside the published demo folder; or
`--telemetry_dir`, `''` for none) for the same questions about people: each
turn's move, how it was made, the seconds taken, and what the player looked at
before choosing (the move sheet, each card read for a moment, folds opened,
plans and armed previews; the page reports them to `/api/look`).

```bash
python -m examples.deep_time.push_audit --write
```

Measures what each Steward move that changes the climate does on every world (at
tick 6 and halfway through, four seeded worlds, four turns against waiting) and
writes the words shown beside the moves to `data/pushes.yaml`. Rerun it when the
physics changes: `push_audit_test` fails while the file is stale. A scenario may
still word a push itself (`pushes:`) or scale a move to its world (`effects:`,
e.g. Great Acceleration's finer basalt dust). It also keeps each move's
turn-by-turn change to the temperature (`profiles:`), from which the Steward's
forecast reckons its late readings forward (`forecast.py`).

## Scenarios

| Scenario             | Goal                      | Fail                      |
| -------------------- | ------------------------- | ------------------------- |
| `daisyworld`         | on a 48-region planet (6  | 10 ticks of band stress   |
:                      : latitude bands × 8        : (each tick out of band    :
:                      : sectors, energy balance   : adds one, each tick in    :
:                      : with heat diffusion and   : band heals a half), or    :
:                      : ice), keep the global     : all daisies dead          :
:                      : temperature in 10–35 °C   :                           :
:                      : while sunlight rises 40%  :                           :
:                      : and plateaus past what    :                           :
:                      : daisies regulate alone;   :                           :
:                      : white and black daisy     :                           :
:                      : species choose behaviours :                           :
:                      : each tick and evolve      :                           :
:                      : through slow adaptation   :                           :
:                      : minds                     :                           :
| `great_migration`    | the ice age ends under a  | 8 ticks of band stress,   |
:                      : strengthening sun; bison  : or fewer than 8 bison or  :
:                      : follow their pasture as   : 1.5 wolves (herd sizes)   :
:                      : it shifts, bold wolf      : at tick 40                :
:                      : packs follow the bison    :                           :
:                      : and geese pass over. Cool :                           :
:                      : the world with bright     :                           :
:                      : clouds, sow grass, and    :                           :
:                      : thin the packs near the   :                           :
:                      : herds you choose before   :                           :
:                      : they eat the herds out    :                           :
:                      : and starve; the map draws :                           :
:                      : each herd's tracks        :                           :
| `herds`              | Daisyworld with a grazing | 10 ticks out of band, or  |
:                      : food web\: grazers evolve : all daisies dead          :
:                      : which petals they eat,    :                           :
:                      : warming or cooling the    :                           :
:                      : planet (the map shows     :                           :
:                      : them as flocks, pale or   :                           :
:                      : dark by their diet, with  :                           :
:                      : their tracks); the        :                           :
:                      : Steward may thin the      :                           :
:                      : herds                     :                           :
| `snowball`           | a frozen ocean world      | still frozen at tick 22   |
:                      : under a faint sun         :                           :
:                      : (geologic era, 5 Myr      :                           :
:                      : ticks)\: rouse the        :                           :
:                      : volcanoes and watch the   :                           :
:                      : air gauge; once the air   :                           :
:                      : holds enough carbon       :                           :
:                      : (×1.3–2.0, shown on the   :                           :
:                      : gauge), a dark haze tips  :                           :
:                      : the ice, but not under    :                           :
:                      : volcanic ash              :                           :
| `great_oxidation`    | 2.6 billion years ago (20 | 6 ticks out of band, or   |
:                      : Myr ticks), methanogen    : oxygen or aerobes short   :
:                      : mats keep a faint sun's   : at tick 40                :
:                      : planet warm with methane  :                           :
:                      : while cyanobacteria make  :                           :
:                      : oxygen that iron soaks    :                           :
:                      : up; when the iron sinks   :                           :
:                      : fill, oxygen poisons the  :                           :
:                      : anaerobes and burns the   :                           :
:                      : methane away. End with    :                           :
:                      : oxygen at 10% of today's  :                           :
:                      : level, aerobic mats on 5% :                           :
:                      : of the planet and the     :                           :
:                      : climate in 12–40 °C,      :                           :
:                      : through the cold snap     :                           :
:                      : when the methane burns    :                           :
:                      : away. Washing iron into   :                           :
:                      : the seas delays the       :                           :
:                      : crisis, seeding           :                           :
:                      : cyanobacteria hastens it, :                           :
:                      : seeding aerobes inherits  :                           :
:                      : it                        :                           :
| `fire_and_iron`      | three peoples with        | a people loses three      |
:                      : different land (farmland, : fifths of its population, :
:                      : ore, forest), entities on : 8 ticks of band stress,   :
:                      : the contrib               : or too little iron        :
:                      : async-deliberation        :                           :
:                      : prefab, farm, cut forests :                           :
:                      : for charcoal, mine and    :                           :
:                      : smelt iron on a temperate :                           :
:                      : world (25-year ticks),    :                           :
:                      : shifting their labor,     :                           :
:                      : learning techs and        :                           :
:                      : trading on a market each  :                           :
:                      : generation; felling       :                           :
:                      : forests pales the land    :                           :
:                      : and kilns warm the air;   :                           :
:                      : the Steward can replant   :                           :
:                      : forests. Forge 110 iron   :                           :
:                      : by tick 40 with the       :                           :
:                      : climate in 5–30 °C        :                           :
| `light_delay`        | colonists on a cold world | fewer than a quarter of   |
:                      : four light-years from     : the colonists left, or    :
:                      : home (1-year ticks) mine  : the sky still below 0 °C  :
:                      : ice, grow food under      : at year 40                :
:                      : glass, split water for    :                           :
:                      : air, tend solar farms and :                           :
:                      : run terraforming plants;  :                           :
:                      : warm the sky from −60 °C  :                           :
:                      : to 0 °C by year 40 with   :                           :
:                      : at least as many          :                           :
:                      : colonists as landed,      :                           :
:                      : while the sky radiates    :                           :
:                      : warming away. Power is    :                           :
:                      : capped by solar panels,   :                           :
:                      : which wear out and only   :                           :
:                      : home can teach them to    :                           :
:                      : make\: Mission Control    :                           :
:                      : reads the colony four     :                           :
:                      : years late and its        :                           :
:                      : designs and directives    :                           :
:                      : arrive four years after   :                           :
:                      : they are sent             :                           :
| `archipelago`        | a warm sea of seven       | 8 ticks of band stress,   |
:                      : islands with life on only : too few islands green at  :
:                      : one\: plants spread over  : the end, or all plants    :
:                      : an island by themselves   : dead                      :
:                      : but cross the sea only in :                           :
:                      : the crops of migrating    :                           :
:                      : geese or by the Steward's :                           :
:                      : sowing, one island at a   :                           :
:                      : time (unaimed it reaches  :                           :
:                      : the nearest bare island;  :                           :
:                      : aim it from the phone to  :                           :
:                      : choose); have 85 in 100   :                           :
:                      : of the islands green      :                           :
:                      : (plants on 45% of their   :                           :
:                      : land) by tick 32, through :                           :
:                      : a volcanic winter and two :                           :
:                      : blights; carrying seed    :                           :
:                      : costs no omega but loses  :                           :
:                      : some                      :                           :
| `savanna_fire`       | a hot, dry world of grass | 8 ticks of band stress,   |
:                      : and scattered woods where : or the woods shrink below :
:                      : lightning strikes ten     : nine-tenths               :
:                      : times as often and fires  :                           :
:                      : take nine-tenths of the   :                           :
:                      : woods they reach; summon  :                           :
:                      : rain over the country you :                           :
:                      : choose (soaked ground     :                           :
:                      : will not burn for a       :                           :
:                      : while) and replant, while :                           :
:                      : bison, wolves and geese   :                           :
:                      : roam; keep the forests on :                           :
:                      : nine-tenths of the land   :                           :
:                      : they held at the start    :                           :
:                      : until tick 40             :                           :
| `sandbox`            | every system at once (50  | no seed ship by tick 50,  |
:                      : ticks of five years)\:    : a people loses three      :
:                      : three peoples of the      : fifths of its population, :
:                      : electric age with coal,   : or 8 ticks of band stress :
:                      : treaties, the Gazette,    :                           :
:                      : Gaia's backlash, trade,   :                           :
:                      : automation, plans and     :                           :
:                      : rocketry, on a living     :                           :
:                      : world tended by the       :                           :
:                      : Steward under a           :                           :
:                      : brightening old sun (keep :                           :
:                      : it cool where grass and   :                           :
:                      : forest thrive; protect    :                           :
:                      : the woods from the axes). :                           :
:                      : Win by launching a seed   :                           :
:                      : ship (three rockets, two  :                           :
:                      : solar panels and ten      :                           :
:                      : seeds gathered from       :                           :
:                      : vaults that fill faster   :                           :
:                      : on a diverse world) while :                           :
:                      : biodiversity is at least  :                           :
:                      : 0.17                      :                           :
| `great_acceleration` | industrial peoples        | a people loses three      |
:                      : (5-year ticks) who know   : fifths of its population, :
:                      : tiers 1–2 discover coal\: : CO₂ above 1.3 times the   :
:                      : coal mines, steam engines : start, or the climate     :
:                      : (energy multiplies        : outside 5–25 °C           :
:                      : labor), steel and         :                           :
:                      : railways; coal burns into :                           :
:                      : CO₂ and soot, and         :                           :
:                      : pollution makes the land  :                           :
:                      : strike back. Make 600     :                           :
:                      : energy by tick 40; coal   :                           :
:                      : treaties (diplomacy) cap  :                           :
:                      : each signatory's coal,    :                           :
:                      : the Cedar league's        :                           :
:                      : Gazette sways every       :                           :
:                      : people's consent for      :                           :
:                      : coal, and the clean way   :                           :
:                      : out is solar power, built :                           :
:                      : by planning whole chains  :                           :
:                      : at once (research runs    :                           :
:                      : 1.8× in this industrial   :                           :
:                      : age)                      :                           :

## How it fits Concordia

Everything runs on the stock `prefabs/simulation/generic.Simulation` and the
stock `Simultaneous` engine. Deep Time adds no engine.

*   **World GM** (`prefabs/world_gm.py`): `switch_act.SwitchAct` with the
    well-known component keys, so the game master makes no language model calls.
    `WorldStep` (a `NextActingAllEntities` subclass) runs the deterministic
    world pre-step when the engine asks who acts next; `ActorResolution` turns
    structured actions into world changes; `terminate.Terminate` ends the game;
    `make_observation.MakeObservation` delivers queued observations.
*   **Checkpoints**: the standard way. `sim.make_checkpoint_data()` saves each
    entity's and game master's components, and the topic bus and the sim-time
    scheduler travel with them: every entity carries the contrib
    `TopicBusComponent` first (as the contrib async-deliberation entity does),
    the World GM's `TopicObservations` and `NumericClock` hold the bus and the
    clock, each slow mind is a `SimTimePublisher` with its work in flight, and
    each transport keeps its deliveries on the road and its random draws.
    `WorldStep` carries the world itself (`GameContext.get_state`: the planet,
    queued observations, attention, wake-ups, standing orders).
    `simulation.load_checkpoint(sim, context, data)` restores a new simulation
    of the same scenario and seed, founding again any species born in play
    first; a restored game plays on exactly as the live one
    (`checkpoint_test.py`). Listeners outside components (wake-ups, slow-mind
    answers, standing orders) ignore the bus's replay while it is restored.
*   **Steward** (`prefabs/steward.py`): the upstream `minimal__Entity` prefab
    with topic-bound `pubsub.Subscriber` instruments as `extra_components`.
*   **Instruments** (`perception.py`): every belief is a topic-bound
    `Subscriber` whose stream runs through contrib `operators.with_age`, so it
    reads e.g. `[tick 7, 2 ticks old]`; news digests merge several topics. The
    Steward's instruments also run through contrib `operators.transport`: its
    instrument tier (`data/instruments.yaml`: standing stones, naturalists,
    orbital radiometers) sets how late readings arrive, how many are lost and
    how noisy temperatures are, and it can spend Ω to upgrade. The report plots
    what the Steward read against the truth (the epistemic gap).
*   **Attention** (`attention.py`): the Steward can watch only as many of its
    four instruments (thermometer, census, air gauge, tide gauge) as its
    instrument tier allows (2, 3, 4). An unwatched instrument's `Subscriber`
    receives nothing, so its belief ages, and shows one short line; watching one
    instrument instead of another costs the Steward its turn but no Ω.
*   **Wake-ups** (`wakeups.py`): in scenarios with `wakeups: true` a species
    deliberates only when an `InterruptMask`-matched topic carries news
    (eruptions, diplomacy addressed to it, or a `bio.alerts.<key>` alarm that a
    contrib `deadband` reflex raises when its meadows' temperature or cover
    moves) or when its 4-tick `Timer` expires. The masks and timers live in the
    upstream `interrupt_scheduling.EntityScheduler`; bus messages become its
    events through contrib `topics.as_interrupt_event`. Daisyworld spends about
    40% fewer choice calls this way.
*   **Advisor**: the Steward's slow mind, a
    `Publisher(QuestionOfRecentMemories)` writing briefings to `gaia.briefings`
    two ticks after it is asked. It alone reads an almanac
    (`components/advisor_mind.py`): the sun five ticks ahead and the eruptions
    the geologists expect, exact because the seeded eruption stream is stepped
    ahead on a copy.
*   **Chronicler** (`prefabs/chronicler.py`): stock memory and observation
    components, a chronicle `Publisher` that lands a chapter five ticks after it
    starts, and a `PuppetActComponent` so acting costs no language model call.
*   **The map** (`world/tiles.py`, `world/biomes.py`, `world/tilelife.py`,
    `world/fire.py`): 1,200 tiles refine the 48 regions; climate layers set 13
    biomes; plants grow and spread tile by tile; lightning fires run through
    dry, thickly grown country, burning carbon into the air and leaving ash;
    bison herds graze tile by tile and wolf packs hunt them, fish schools follow
    the fisheries' stock and flee dead zones, and geese migrate with the
    seasons, sowing grass where they land (`world/herds.py`). Peoples build
    towns that work the land around them, joined by roads and trade routes
    (`world/settlements.py`); their smog spoils nearby fields and their runoff
    opens dead zones off their own coasts (`world/pollution.py`). In Light Delay
    the colony's own world, New Ararat, thaws on its own tile map
    (`world/ararat.py`). The phone draws it at 8 pixels a tile, with eight
    overlays (Seasons plays a year: snow and sea ice following each hemisphere's
    winter) and gliding herds.
*   **Seed banks** (`world/seedbank.py`): spreading spends a species' seed bank,
    holding and dormancy refill it; an empty bank exhausts a spreader, and a
    nearly vanished species sprouts again from a full enough one.
*   **Symbiosis** (`world/symbiosis.py`, `components/pact_mind.py`): species
    offer mixed-meadows pacts on `diplomacy.<to>.<from>`; the partner's pact
    mind is a `Publisher(..., trigger_on_observe=True)`, so the offer wakes it
    at once, and its answer lands two ticks later.
*   **Speciation** (`world/speciation.py`): a species whose range spans very
    different climates splits; the slow model names the newcomer and the World
    GM adds it mid-run with `Simulation.add_entity`.
*   **Grazers** (`world/fauna.py`, `prefabs/grazers.py`): a herbivore clade with
    a Holling type II response per region, migration, behaviours and its own
    adaptation mind; its evolved diet decides which petals it crops.
*   **Species** (`prefabs/species.py`): also `minimal__Entity`, plus a
    genome sheet (the World GM publishes it each step on the species' own
    topic; a stock `Subscriber` reads it) and a slow adaptation mind:
    `pubsub.Publisher(component=QuestionOfRecentMemories(...),
    skip_if_busy=True)` scheduled on a contrib `LatencyScheduler`, so its answer
    lands exactly three ticks after it was triggered. Answers go to
    `bio.species.<key>.adaptation`; the species enacts one by its own action,
    which the World GM resolves into bounded trait changes
    (`world/evolution.py`).
*   **Carbon cycle** (`world/carbon.py`): six reservoirs on the contrib
    `StockFlowLedger`; outgassing, silicate weathering, ocean exchange,
    photosynthesis, decay, burial and subduction are audited transfers, and
    atmospheric CO₂ sets the greenhouse forcing of the climate.
*   **Civilization** (`world/civ.py`): polities on the same contrib ledger (one
    holder each) eat grain, grow or starve, and split their labor over six
    ledger `Recipe`s: farm (yield from the local climate), cut wood (from the
    forests, which it clears), burn charcoal, mine ore (seeded deposits that
    eruptions refresh), smelt iron in bloomeries and forge tools (which raise
    every worker's output). Kiln and bloomery carbon goes to the carbon cycle.
*   **Technology** (`data/techs.yaml`): twelve tier 1–2 techs bought with
    research (a ledger resource made by a `study` recipe). Their effects are
    data: more output from a recipe, fewer inputs, less forest cleared, less
    grain spoiled, faster research, or unlock tags for logistics and reflexes.
*   **Energy** (tier 3): coal seams, coal mines, steam engines and steel mills;
    energy drives machines (up to 1.8× the work per person), and coal puts CO₂
    and soot into the air.
*   **Tier 4**: glass, printing, cement, copper and wire, electricity,
    fertiliser from the air, solar panels and smoke scrubbers (30 techs and 23
    recipes in all, as data in `data/techs.yaml` and `data/civ.yaml`).
*   **Gaia's backlash** (`world/backlash.py`): industry pollutes the regions a
    people lives in; forests absorb it; what lingers raises pressure until the
    land strikes with pests or plague. Restless lands show on the Steward's
    Omega meter, and cleansing rains calm them.
*   **Plans** (the ratio planner): a people that knows mathematics can put 40%
    of its labor into a whole production chain in one decision ("plan a balanced
    chain to build solar panels"); the contrib `stock_flow_ledger.balance_chain`
    works back from the target through every input's producer, for that people's
    own land, so nothing upstream is short or piles up. Chains run on solar
    power once a people has panels.
*   **Rocketry** (tier 5): rocket fuel, rockets (a plannable chain) and
    launches; weather satellites sharpen their owner's harvests and feed Gaia's
    instruments at the best tier.
*   **The seed ship** (tier 5, interstellar arks): seed vaults yield with the
    planet's biodiversity; the ship (rockets, panels, seeds) is one balanced
    plan; it wins the Sandbox only if it leaves a living world.
*   **The colony** (`world/colony.py`): a world light-years away with its own
    contrib `Ledger`, six colony recipes and the ratio planner; colonists eat
    and breathe, shortfalls kill, terraforming warms a sky that cools back.
*   **Fisheries**: fishing boats let a people fish the one shared stock of the
    seas (a commons that regrows logistically); the catch falls as sea ice
    spreads or as coal's carbon sours the seas. Peoples with diplomacy can sign
    fishing quotas, a second kind of treaty.
*   **Dead zones**: fertilised fields (industrial chemistry) wash runoff down
    the rivers; the nutrient load opens coastal dead zones, wider in warm seas,
    where fish neither live nor breed, so one people's fertiliser empties
    everyone's nets. The seas flush the load away slowly; wetlands and buffer
    strips catch most of it. Peoples' sheets, the Steward's tide gauge and
    `ocean.alerts` report them; emptied or choking seas weigh on biodiversity,
    and the Steward can restore the coastal marshes. Peoples with diplomacy and
    chemistry can sign runoff pacts, a third kind of treaty (each kind's wording
    and metric is data in `civ.yaml`).
*   **Price news**: market prices reach a people by courier two clearings late
    until it has the telegraph; its orders are priced on what it believes
    (`orders_on_stale_prices` in `summary.json`).
*   **The power grid**: with the power grid tech, energy is a tradable good
    carried by wire (at once, 10% lost); the trade menu leads with the biggest
    surpluses.
*   **Coal treaties** (`world/treaties.py`): a people that knows diplomacy can
    offer another a coal treaty (each burns at most 4 coal a generation for ten
    generations). The partner's treaty mind (a `Publisher` with
    `trigger_on_observe`) answers in language; burning past the cap breaks the
    treaty and embargoes trade between the two for six generations. The record
    of treaties, embargoes and breaches is public on `civ.treaties`.
*   **The Gazette** (`world/media.py`): a contrib `forum.ForumState` every
    people reads. With the printing press and a sheet of paper a people can
    print a broadsheet, written by its press (a slow mind), for clean air or for
    coal and progress. Readers judge it against the Gazette's sky watch and vote
    (forum karma is credibility); credible broadsheets sway every people's
    consent for coal, and coal mines yield in proportion to it. A people that
    burns coal can pay the coal guilds for pamphlets defending it (half the sway
    of a free press).
*   **Trade and logistics** (`world/trade.py`): peoples post standing orders
    (offer a surplus, bid for what their land lacks) that clear on the contrib
    `MarketPlace`; every sale ships on a stock-flow ledger link (porters, or
    faster carts once a people knows roads) and arrives ticks later, while
    silver changes hands at once. Prices are published on `econ.prices`.
*   **Automation** (`world/automation.py`): once a people knows granaries it can
    adopt standing orders ("sell ore whenever we hold more than 6") as contrib
    reflexes on the census; they place market orders with no language model
    call; each people's scribes (a slow mind) can also write them as reflex
    JSON, which is sandboxed (public inputs, own orders only, at most four)
    before it is installed. The report shows the share of orders placed by
    reflexes.
*   **Oxygen and methane** (`world/oxygen.py`): air and crust stocks on the same
    ledger. Each species has a metabolism (anaerobic, oxygenic, aerobic); iron
    soaks up oxygen until it runs out, methane burns oxygen, and free oxygen
    poisons anaerobes, feeds aerobes and removes the methane greenhouse.
*   **Ocean and geology** (`world/ocean.py`, `world/geology.py`,
    `world/geography.py`): seeded continents, sea ice, sea level and pH; seeded
    eruptions with ash, outgassing and fertile soil.
*   **Topics and time**: contrib `pubsub/topics.TopicBus` and
    `pubsub/sim_time.SimTimeScheduler` (library PR).
*   **Observations**: contrib `TopicObservations` on the World GM turns topic
    messages into observations for every entity whose access list (topic
    prefixes) matches: the Steward hears geology, biology and ocean alerts;
    species also hear the Steward's omens; the Chronicler hears everything.
*   **Time**: contrib `NumericTimeModel` / `NumericClock` (library PR), so ticks
    can be 50 million years long.
*   **Models**: `language_model_setup` plus `RetryLanguageModel`,
    `CallLimitLanguageModel` and `ProfiledLanguageModel`.

## File tree

```
examples/deep_time/
├── README.md
├── run.py                  # CLI; writes the run folder
├── simulation.py           # builds Config and Simulation for a scenario; load_checkpoint
├── game.py                 # GameContext shared through prefab params
├── actors.py               # per-role rules: action specs and resolution
├── perception.py           # instruments: ageing beliefs, transports, news digests
├── attention.py            # attention slots: which instruments the Steward watches
├── forecast.py             # the Steward's forecast: dead reckoning from late readings
├── crises.py               # crises with countdowns, from a role's own readings
├── wakeups.py              # event-driven species: interrupt masks, timers, alarms
├── audit.py                # complexity audit (data/systems.yaml)
├── sweep.py                # balance sweeps over seeds and mock variants
├── model_setup.py          # mock / Together / Ollama models by role
├── mock_language_model.py  # deterministic, game-aware offline model
├── data/                   # YAML: physics, topics, interventions, scenarios, civ, techs,
│                           #   trade, backlash, media, systems (the audit registry)
├── world/                  # deterministic substrate: grid, climate, Daisyworld, evolution,
│                           #   speciation, symbiosis, fauna, geology, ocean, carbon, oxygen,
│                           #   planet, choices (labels ↔ rules), civ (polities, techs), trade,
│                           #   automation (reflexes), backlash, treaties, media (the Gazette),
│                           #   seed banks, tiles (the 1,200-tile map), biomes (climate layers),
│                           #   tilelife (plants and mats on tiles), fire (wildfire, scars, ash),
│                           #   herds (herds and packs with positions), settlements (towns, roads), pollution, ararat (the colony's world), news,
│                           #   publishing (topic payloads and renderers), colony (Light Delay)
├── components/             # World GM components (world step, specs, resolution); slow minds:
│                           #   pact and treaty minds, policy compiler, press mind, the
│                           #   printers' relay mind (omens retold), each people's take
│                           #   mind, the Advisor's mind (with its almanac)
├── prefabs/                # World GM, Chronicler, grazers; Steward, species, polity, colonist
│                           #   and Mission Control configs; the playable entity (a person or
│                           #   a model)
├── web/                    # play or watch from a phone: session (HumanInput reader), server,
│   │                       #   page
│   └── js/                 # the page's script modules (inlined when served) and their
│                           #   tests (`node --test`, run by web/js_test.py)
├── report/                 # self-contained HTML run report
├── scenarios/              # one module per scenario
└── *_test.py               # unit tests and mock end-to-end tests
```

## Tests

```bash
pytest examples/deep_time
```
