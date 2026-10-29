# Lifeways

SimLife as artificial social science. A small 2D world where people are born,
work, find partners, raise children and die, generation after generation, for
as long as you let it run. What evolves is culture. A person's **customs** are
components of that person's own Concordia entity. Like plasmids, they are
inherited at birth (copied, or recombined from two parents, domain by domain),
offered by others, taken up or dropped, and mutated as they are copied.

## Quickstart

Offline, with the deterministic mock model:

```bash
python -m examples.lifeways.run --years=200 --soak
```

With live minds (each person-year is one call), from a local Ollama, or a
model Concordia's contrib wrappers support (the Together key from
`TOGETHER_API_KEY`); `--settings` adds settings as JSON, e.g. a plan of
interviews:

```bash
python -m examples.lifeways.run --model=ollama:llama3.2:3b --years=6 --founders=6
```

```bash
python -m examples.lifeways.run --model=together_ai:deepseek-ai/DeepSeek-V4.1-Flash --years=20 --settings='{"plan": {"6": "ask Kaka: What do you live by, and why?"}}'
```

The summary counts the calls and the tokens (estimated), and keeps the
interviews' transcripts.

A mind may say more than its move; the world reads the first move offered
that the answer names, and credits only the customs it once gave the one who
answers.

The run writes these files to `--out_dir`:

| File | Contents |
|---|---|
| `census.jsonl` | a row a year: the world's census (population, births, deaths, moves, how many moves named a custom (`cited`), the first answers that named no move (`unread`), sowings that grew nothing for naming no know-how (`barren`, with the first answers), how unequal food stores are and how much was shared (`gini`, `shares`, `fed`, `judged`), how often norms were kept, seen and unseen (`norms_kept`), offers by channel, variants, the skill people work by) and, under `held`, what the living hold (read from their entities) |
| `dead.jsonl` | each person who died: their name and their entity's final state |
| `customs.jsonl` | each variant of a custom once nobody living holds it, with its parent: the long record of culture |
| `summary.json` | the run's totals, with memory use and seconds per year (`--soak`) |

## The page

```bash
python -m examples.lifeways.web.server --port=8921 --base_path=/lifeways
```

A phone-first page plays a running world. It draws the valley, the
peoples and the places, and the census as graphs. Tap a person to see
their family over three generations, the dead included as the valley
remembers them (`World.dead`). Their customs are listed too, with how each
came and from whom; a person's `Learner` remembers both. Each custom opens
how it has changed: the versions people hold now, most held first, with
who holds each and how they came by it. Folded below are the versions kept
only in libraries or shrines, the family tree from the oldest version
remembered, and the dead branches (counted, the latest listed). The page
remembers the variants the world has forgotten, up to a bound. Every part
of the page says what it is in words: no ids, no settings' names. The
person on the page is the
Experimenter: through the stock `HumanActComponent` and a `PageInput`
adapter, each year it answers with the move the person queued (a release
from the custom designer, or stopping the world), or else the person's
standing choice, watching, at the pace the person sets. The server only
builds, plays and reads the simulation (`web/session.py`); the run ends
when the Experimenter stops the world.

The page plays MOCK minds by default. `--model=<api>:<model>` gives people
live minds through the contrib model setup, e.g. `--model=ollama:llama3.2:3b`
(Together reads `TOGETHER_API_KEY`), and the footer says so. Live minds are
slow: on a laptop llama3.2:3b takes about 45 s a year for 14 people. In tick
mode a year waits for every answer. With `--free=60` a year also closes
after 60 s, and a late answer is dropped. At 15 s nobody's answer lands in
time.

With `--save=FILE` the world outlives its server.
- **When it saves.** The session saves the stock checkpoint
  (`Simulation.make_checkpoint_data`) with the page's census rows, pace,
  notebook and followed releases:
  - at a year's close, about once a minute;
  - at the close of the year the person stops the world;
  - when the server is stopped (SIGTERM) at the next year's close, or at
    once if the world is paused or stopped. No year closes then, so the
    snapshot is taken in the World GM's lock, mid-year.
- **How it saves.** The snapshot is taken where the world is whole; a
  thread of its own writes the file, so the world does not wait for the
  disk.
- **Restoring.** A new server of the same scenario, seed and overrides
  restores the save (`simulation.load_checkpoint`) and plays on from that
  year. The settings are the new build's. The restore is checked: the
  stock restore logs a component that fails and goes on, so the session
  compares the World and the people with the save.
- **Setting aside.** A save of another world, or one this build cannot
  restore as it was, is set aside (`FILE.set-aside`), and a new world
  begins. A stopped world stays stopped; "Begin a new world" sets it aside
  (`FILE.stopped`) and starts afresh.
- **Logs.** At every close the engine logs the Experimenter's `Closed` as
  an error, with a traceback: that is how a page run ends. The server
  sends the engine's prints to the null device for its whole life.

**Your releases.** A person's releases are recorded in the census rows
(`releases`: id, how, where, how many heard a song), and the page follows
each one. Year by year it counts how many living people hold the release
or a variant descended from it (their Learners; measurement). A release
lives on while it, or a variant of it, is held or kept in a place. The
first year neither is true, it has died out, and the notebook says so. A
ring on the map marks where the next release goes, with how many would
hear a song there. A newcomer can see a released custom spread or die
within minutes at the default pace.

**The notebook** (DESIGN §9) is the person's lab record, kept by the
session and saved with the world. It holds the hypotheses, observations
and notes the person writes, and each move the person makes. The World's
own account of each move is there too: the census row's `reports`, what
the World told the Experimenter. So are changes a scientist would note:
the number of peoples or tongues, a custom becoming the most widely held
for the first time, few or none left. `GET notebook.md` gives a copy as
Markdown.

**Data to keep.** The census section links three CSV files, for analysis
outside the page: `data/census.csv` (the census rows kept, a column for
each number, nested ones spread out), `data/people.csv` (a row for each
living person and custom they hold now: the person as the World knows them,
their people by size, 0 the largest, and the custom as their Learner keeps
it, read by the harness, never by the GM) and `data/surveys.csv` (each
survey answer kept, with the customs it named).

**Live as a founder.** With `--play_as`, or "live as a founder" when a new
world begins, the person also lives in the valley as its first founder.
That founder is an ordinary person of the valley built from the same prefab
(`Person`); only its acting component differs. The prefab takes an
`act_component_factory`, and the session gives this one a `PlayedAct`
(`components/played.py`): the stock `HumanActComponent` over a `PersonInput`
adapter, and the person's habit for when nobody answers. The World asks
the person what it asks everyone, through its game master:
- each year's move, with the customs the person follows in doing it;
- which customs to pass on to a child about to be born;
- the Experimenter's interview questions.

The page's "Your life" panel shows the question, how the person's life
stands (its status line, as it observed it), its moves and customs to
choose from, and all it knows and remembers. Choosing a move ticks the
know-how the World counts for it: food skills for gathering, farming
know-how for sowing (without it nothing takes root), reading for the
library. Those who see the person may take up the customs it follows. A
dashed white ring marks the person on the map.
- **Waiting.** While the person decides, the year waits. The input never
  makes up a move: a question unanswered for two minutes of play (pause and
  it waits) raises `Unanswered`. The person's acting component then acts by
  its own habit, logged as habit rather than as the player's: it gathers
  food, passes on the customs its `Learner` holds, or keeps its counsel.
- **Endings.** Stopping the world hurries the input, so nobody waits.
  The person dies as anyone does, and the valley plays on.
- **Living on.** Once the played person dies, the player is born again
  (`World.heir`), as the first child born (never a newcomer):
  - of its line, a descendant (its own child, if one is born the year it
    dies);
  - or, once nobody of its line lives, of its people;
  - or, once none of its people live either, of anyone.

  The World GM's `TurnGate` keeps the lives played. The heir joins through
  the stock roster as a second prefab of the Config, the `Person` with a
  `PlayedAct` (`PLAYED_PREFAB`). So a save restores the heir as
  played, like any other joined player.
- **Saves and twins.** A played world's save restores only as played. In a
  twin, minds play the person, and each heir.
- **Not in free mode.** A played person's year waits for the player, so a
  person cannot be played where years close on their own (`--free`).

**Fork a twin.** The person may fork a twin of the running world. At the
next year's close the session takes the same checkpoint and builds a second
simulation of the same world from it, with one lever pulled
(`scenarios.with_lever`). A lever is a world rule the World reads from its
settings as it plays: `richness`, `regrowth`, `seed_bank`, `subsistence`,
`mutation`, `dialect_drift`, `garble` or `child_cost`, within bounds, or
`without` a channel. Both worlds play on side by side.
- **The twin follows its world.** Its Experimenter (`TwinInput`) waits for
  the world's Experimenter's answer for each year, never running ahead.
  It makes the same move in the same year if the move shapes the world (a
  release, a disaster, migrants, an institution), and otherwise watches.
  So the lever stays the one difference.
- **What stays the world's.** Questions and stops reach the first world
  only, and the person's pace reaches both.
- **The page** draws the twin's census dashed over its world's, and says
  how each release fares in the twin. "The twin, side by side" reads both
  censuses over the years both have lived since the fork: each measure's
  mean a year in each world, and how much the twin differs, the largest
  differences first (the page's own reading of the two censuses).
- **Exactness.** With no lever changed, a twin's census is its world's,
  year by year, moves and all (tick mode is exact).
- A twin is not saved.

## Experiments

### What spreads when released

MOCK Sandbox, 6 seeds, build ee4a3a5f (every MOCK run has differed since a
person's answer began with their name, f0eb8b80). In each, the Experimenter
sang a custom in year 14
where most would hear it, and the release was followed to year 85 as the
page follows it (`Session.releases`). A short harness over the page's
session made the table; it is not part of this example.

| Released | alive at year 85 | most holding it at once (range) |
|---|---|---|
| a habit (food) | 6/6 | 33–55 |
| a rule (sharing) | 6/6 | 25–45 |
| whom to learn from | 6/6 | 17–50 |
| a marriage rule | 4/6 | 0–11 |
| a value; a belief | 2/6 each | 0–5 |
| know-how, level 1 / level 4 | 1/6 / 3/6 | 0 / 0–43 |

Each song was heard by 2–7. Ways people act on in their everyday moves are
seen and copied, and spread; beliefs and values released the same way
mostly died out, though births and teaching can carry any kind (why they
fare worse is not settled). Know-how is taken up only if it is better than
what one knows.

### A property norm, read by two kinds of mind

First Farmers, 6 founders, 24 years; the norm "We take nothing from
another's field." held by every founder, against none. MOCK minds over 20
seeds; live minds (Ollama `llama3.2:3b`, the stock framing and answer
prefix) over seeds 0-2 (build 272aa666):

| Minds | Where the norm sits | "sow a field" | fields grown | sowings that grew nothing | spared others' fields |
|---|---|---|---|---|---|
| MOCK | none / farming | 15.3 / 16.0 a run | | | 0 / 4.3 a run |
| live | none | 88 | 73 | 17% | 20 |
| live | farming | 72 | 32 | 56% | 11 |
| live | authority | 87 | 62 | 29% | 44 |

- The rule-following mock does what the norm says: it spares others'
  fields and sows as before.
- Live minds holding the norm in the farming domain try to sow about as
  often, but their sowings grow nothing more than half the time. A
  sowing makes a field only if the answer names farming know-how the
  sower was given.
- Mostly they no longer hold it: of 20 barren sowings (seeds 1-2, build
  82e63b00), 16 were by people who had lost the farming know-how
  (`barren_knowing` in the census). The norm comes in two forms (expected
  of people, held as one's own), and both take the farming domain's two
  places, so whoever takes it up loses the know-how. The World, which
  never reads minds, still offers sowing to anyone it once gave the
  know-how, and nothing grows. About as many hold the know-how in both
  arms, but they are not the ones who sow.
- The same norm in another domain (`authority`) takes no farming place:
  it barely hurts sowing and doubles sparing.
- So what a custom does depends on the minds that act on it, and on
  where it sits among their customs. Over seeds 0-4 the farming-domain norm
  cut the fields grown in all five seeds (130 against 69).

### The experiment tool

```bash
python -m examples.lifeways.experiment --lever=richness=0.5 --seeds=5 \
    --years=200 --out_dir=/tmp/richer
```

A twin on the page compares one world with itself, one lever changed. An
experiment repeats that over seeds.
- **The runs.** For each seed it plays the control (the scenario's
  settings) and the treatment (`scenarios.with_lever`), in MOCK tick mode,
  in processes of their own. Runs are exact, and they do not depend on
  Python's hash seed, so every difference is the lever's or the seed's.
- **The comparison.** A run's measure is its mean over its later half:
  the living, distinct customs held, their diversity, the share holding
  the top custom, peoples, tongues and the skill worked by. Across seeds
  the tool gives the control's and the treatment's mean ± sd, and the
  paired difference: its mean ± sd and in how many seeds it was above or
  below.
- **The output.** It prints a Markdown table and writes `summary.json`,
  `report.md` and every run's census (`census.jsonl`). The harness only
  builds, plays and reads simulations.

## Scenarios

The Sandbox is where everything is, and its presets are settings, not code
(`scenarios.py`; `--scenario=...`):

| Scenario | What it asks |
|---|---|
| `sandbox` | Open Country: the River people found a valley; Hill people may come when it is nearly empty. Everything is on. |
| `plasmid_soup` | Everyone buds alone; customs move freely (people watch from afar, read texts the Experimenter leaves, hear sermons); there are no peoples. Do traditions form without peoples? How fast do customs sweep? |
| `two_valleys` | Two peoples with their own tongues, beliefs and marriage rules, in valleys a ridge parts but for a pass. Ethnogenesis, merger, or both? |
| `first_farmers` | The Sandbox, a third of whose founders know how to sow. Does farming spread or die out, and does the valley come to feed more? |
| `bridged_valleys` | Two Valleys with one founder in each people who marries whom they love. The two bridge the peoples into one; as the custom gains and loses holders, do whole peoples split and merge? |

On the page, a stopped world gives way to a new one of any preset; the save
names its preset, so a restarted server restores the world as it was.

## How it fits Concordia

It runs on the stock `prefabs/simulation/generic.Simulation` and the stock
`Asynchronous` engine: every person has an observe-act loop of its own.
The page shows this live in "How it's built". Each person's panel lists
the person's components, and the section lists the World GM's, part by
part, read from the running entities (`Run.anatomy`, `GET /api/anatomy`).

**People** (`prefabs/person.py`) are entities with these components:
- `AssociativeMemory`, `ObservationToMemory`, `LastNObservations`;
- a `Constant` identity;
- the `Learner` (`components/culture.py`);
- their **customs**, one `Custom` component each.

Customs are added and removed while a person lives (`EntityAgent.add_component`
/ `remove_component`). They are rebuilt on restore through the entity's
`component_factory`.

A person acts through the stock `ConcatActComponent`, or a person's
`HumanActComponent`, or a rule (`act_component_factory`). The stock component
asks the model through an `InteractiveDocument` and begins the answer with the
person's name ("Answer: Kaka "), the sentence Concordia's models are framed to
continue; the World reads the move in it. The `Learner` takes up
the customs offered in observations, by the person's own learning biases:
- each further person seen doing the same is another chance to take it up;
- a kind of custom in a domain is a locus, like a plasmid's incompatibility
  group: a new custom replaces the one the person holds there;
- a skill is taken up only if it is better than the one held;
- past `per_domain` customs of a domain, or `capacity` in all, the oldest goes.

**The World GM** (`prefabs/world_gm.py`) is `SwitchAct` over:

| Key | Component |
|---|---|
| `__next_acting__` | `TurnGate`: each participant answers once a year. The year closes when the last answer is in. |
| `__next_action_spec__` | `Questions` |
| `__resolution__` | `Answers`, which records each answer |
| `__make_observation__` | the stock `MakeObservation` |
| `__terminate__` | the stock `Terminate` |
| `lifeways_world` | the world (`world/`) |

The world is built from the GM's own params; no object is shared with any
entity.

**Closing a year.** Answers are resolved together, in a fixed order, when the
last one is in. A year's outcome therefore never depends on thread timing:
two runs with the same seed are identical. In free mode (the `free`
setting, for live play) a year also closes after so many seconds, whoever
has not answered doing nothing that year: slow minds lag, and runs depend
on timing. Free mode is for short live play. Each dead person's queue is
emptied, but keeps an empty entry in the GM's stock `ObservationQueue`;
removing the entry while other threads take observations would need an
atomic remove in the stock queue.

**Inheritance without mind reading.** Every custom enters the world through
the GM, which keeps the public record of variants (`world/culture.py`). A
parent expecting a child is asked what it passes on, and names custom ids. The
GM copies them, with copying errors, into the newborn's params.

**Copying errors.** A skill copied in error is learned a level lower; a
tongue changes a word. Any other text loses a word, regains one, or gains
or loses its one stress (", always."). Each variant keeps its line's first
text and the words it has lost, at most two. So a custom drifts about its
first wording and keeps returning to it; it never wears away.

Work answers name the customs they follow (`gather food c3 c6`). The world
credits a skill only if the GM once gave it to that person.

**Transmission by enactment.** What people do passes on what they do it by.
The GM sees which customs an answer names and offers copies of them, with the
channel's copying errors, as lines of next year's observations:

| Channel | Who is offered what |
|---|---|
| birth | the newborn: what its parents passed on, recombined by domain |
| upbringing | a child: what its parents do, and what they `teach` |
| observation | anyone within a radius: what the nearest few did it by, with how well fed they are |
| apprenticeship | a youth who chose to `learn a craft`: the skill of the nearest skilled worker, sometimes a level better |
| sanction | one shamed (`shame <name>`): the norm the shamer shames by |
| practice | one who keeps a norm only expected of it: that norm as its own |

**Norms in two forms.** A norm is *internalized* (a person's own, kept when
nobody watches) or *known* (expected by others, kept when someone is within
sight). Children raised in a norm make it their own; so may those who keep a
norm they only know, or are shamed for breaking one. What one sees others
keep, one knows. A known norm never displaces an internalized one.

Whether minds keep them so is measured, not assumed. The World notes, for
each deed, whether anyone was within sight as it was chosen and which norms
it named (`World.norm_acts`); the harness reads which norms each doer holds,
and in which form, from its Learner (`Run.norms_kept`). The census's
`norms_kept` counts, for own and known norms, seen and unseen, how many were
held and how many the deed named; the page draws it as "Norms kept, seen and
unseen". The mock keeps a known norm only when seen, and then names it; a
live mind is told the same things (its status says how many are within
sight; its customs say which are "expected of us") and may not.

**Sharing and shame.** A person with enough may `share food with` a needy
person within sight. One who keeps full stores and gives nothing to the
needy near them is seen doing so; next year, those who saw may `shame` them.
The world reports what was seen; whether it was wrong is each person's to
judge.

Copying errors lose know-how (a skill comes out a level lower) and never add
any; a skill improves only by an apprentice's care. A scenario's `channels`
setting sets each channel's reach, fidelity and the
form a norm comes in by it, and which channels exist. A founding custom's
`held_by` is the share of founders who hold it. What people take up is their own Learners' choice; the harness
reads it for the census (`Run.customs_census`), the GM never does.

**Libraries.** Customs can wait in the world itself, like plasmids in the
environment. One who can read (a skill some founders have, passed on by
teaching) may `found a library`, `write at the library` (exact copies of the
customs named go on its shelf) and `read at the library` (the `reading`
channel offers some of the shelf's texts). Watching someone read teaches
nothing. Shelves outlive their writers, and the world keeps their customs
on record; the census counts the readings that bring back a custom nobody
living has (`rediscovered`). A library unused for long is a ruin, whose
texts can still be read but crumble one by one.

**Schools.** An adult may `found a school` and `teach at the school`: the
customs named become its curriculum, and the teacher keeps the school for a
term. Any child within sight who chooses to `go to school` during the term
is taught the curriculum (the `school` channel), whoever its parents are. A
school that teaches reading makes readers of a whole generation.

**Shrines and rites.** An adult may `build a shrine`. Its rite falls on a
calendar (every fourth year in the Sandbox): then the first who leads the
rite (`lead the rite`) sets the customs it is led by, and all who join it
(`join the rite`) are offered them as their own (the `rite` channel). A
rite nobody leads offers nothing.

**Courts.** An adult may `build a court`. Within its reach (3 steps in the
Sandbox), anyone seen last year keeping full stores while a neighbour went
hungry may be judged there: a grown person nearby may `judge <name>`, naming
the rules it judges by (else the court's law, the rules it last judged by).
The judged is offered those rules to hold as its own (the `sanction`
channel), those within reach see it done and are offered them as what is
expected (`observation`), the court keeps them as its law, and the judged
gives each of the needy it passed over some food (`fine`). One is judged at
most once a year. Where shaming is one neighbour's word, a court is the
valley's: it reaches all who stand near, and it takes food.

**Stories and sermons.** An elder may `tell a story` of the customs it
names to all within sight: the `story` channel retells with many errors, so
stories breed variants. Anyone may `preach` one custom to all within sight:
the `preaching` channel is content-biased, moving beliefs and values more
readily than other kinds, and skills not at all.

**The custom designer.** The Experimenter never edits a mind. It may
release a custom it designed into the world:

```
release <kind> <domain> as a <text or song> at <row>,<col>: <what it says>
```

A text is left on the shelf of the nearest library within sight of the
place (or of a new one there); a song is offered to everyone within sight
of it (the `release` channel). Whether anyone takes it up is theirs to
decide. A scenario's `plan` gives a rule-played Experimenter its moves by
year; on the page a person plays it through `HumanActComponent`.

**Dialects.** Each person speaks a dialect: a list of words, a custom of
its own. A child speaks as its mother does, a word drifting now and then
(`dialect_drift`). Two people understand each other as well as their words
agree, and talk between dialects garbles what passes: the chance of a
copying error rises with misunderstanding (`garble`). The census counts the
tongues spoken and the mean understanding between two of the living.

**Marriage rules and peoples.** One looking for a partner may look for
anyone (`look for a partner`) or court one by name (`court <name>`), among
the singles near them, whom they see as they are seen: how they speak and
which people they descend from (their mother's). Two pair when each would
have the other. A marriage rule is a custom (`marriage`): marry one who
speaks as we do, or only our own people; it shapes whom a mind courts,
never what the world allows. `Run.peoples` measures the peoples: clusters
of the grown whose marriage rules allow each other.

**Work and land.** Whatever a person does with its year, it forages a
little on the way (`subsistence`); gathering is the year's work on top of
it. One who moves looks for better land within a few steps (`roam`). The
land regrows each year, and its richness sets how many it feeds: in the
Sandbox, about 40 (20-60), with hardly anyone arriving.

**Interviews.** The Experimenter may ask a person a question (`ask <name>:
<question>`). The next year the GM puts it to the person before its move,
and the answer goes into the world's transcripts (and to the Experimenter, and
to the page): a mind's tells what the person says ("Kaka says: ..."), a
player's is in their own words.

**Surveys.** The Experimenter may put one question to everyone who can
answer (`survey: <question>`) or to a sample drawn by the year's seed
(`survey <how many>: <question>`). Each is asked it next year as an
interview. The GM files the answers with the survey rather than reporting
each, and as that year closes it tallies which of the customs it gave each
respondent the answer names: by its id, or by two thirds of its text's
words (a mind tells its customs in its own words). The Experimenter is told
the tally ("Most named: ...", and how many named none), and the page lists
it, with the first answers as said. The GM reads only the answers, never a
mind.

**Arrivals.** While the valley is nearly empty, strangers may come (the
`arrivals` setting): in the Sandbox, the Hill people, who keep their elders'
ways, feed only their own and have children young. Their customs spread, or
not, by the same channels as anyone's.

**Disasters and migrants** (DESIGN §2). A disaster strikes within a few
steps of a place (`radius`, three):
- a **drought** leaves a fifth of what grows;
- a **flood** takes what grows on low land (river, coast, marsh), and the
  texts of its places;
- a **fire** takes what grows on grass and in forest, and libraries' texts;
- a **plague** may take each life within reach, the old and the young most;
- a **raid** takes the stores of all within reach, and may kill.

The world sends one by chance (the `disasters` setting: in the Sandbox, one
a century or so). The Experimenter may send one (`send a <kind> at
<row>,<col>`), or bring at least four strangers of the arrivals' people
(`bring migrants at <row>,<col>`). Those within reach are told, the census
row keeps each disaster (`disasters`), and the Experimenter hears what it
took.

**Farming.** The `farming` setting (`yield`, `fallow`) lets one whom the
World gave a farming skill sow the land it lives on (`sow a field`, naming
the skill; hills are never sown). The field then carries `yield` more food
per level of the skill (times the richness) for whoever forages there, and
is its latest tender's; untended for `fallow` years it goes wild. A sowing
that names no skill grows nothing (the sower is told). Where fields others
tend lie within a day's walk, one may gather sparing them (`gather food,
sparing the fields of others`); a dead tender's field is nobody's. Sowing
and sparing are seen and may be copied. Which deeds are seen at all is a
setting too (`seen_deeds`, all by default), for onlookers (observation) and
children (upbringing) alike. No founder of the Sandbox knows how to farm (a
song may teach it); in First Farmers a third do.

**Institutions.** The Experimenter may found a library, school, shrine or
court at a place (`found a <kind> at <row>,<col>`), where the world has that
kind and none stands within its spacing; it stands empty until people use
it.
It may also close one (`close the <kind> at <row>,<col>`): the place and
what it held are gone.

**Partners.** One who looks for a partner looks until paired, with the
nearest who is looking too. Parents and children, and siblings, never pair.

**Births and deaths** go through the World GM's stock `PlayerRoster`,
which holds the list of entities the simulation plays:
- A newborn joins the game (`PlayerRoster.join`, built from the person
  prefab); the engine starts its loop.
- The dead leave it (`PlayerRoster.leave`), which ends their loops; the
  harness's archive is given each (`Lifecycle.departed`).
- Neither the simulation nor the engine is told: both follow the list.

**Indefinite runs stay bounded.**
- The raw log is trimmed (`play(log_retention=...)`).
- The async engine streams its log in batches; each component keeps only its
  most recent logging data.
- The dead leave the simulation, and the variants nobody holds leave the
  world's record (both go to disk); history windows roll.
- The Experimenter, who lives as long as the world, keeps only its latest
  memories (`RecentMemory`).
- `Run.play` sends what the engine prints to the null device (one redirect
  for the process, however many worlds play).

**Checkpoints** are the standard ones (`Simulation.make_checkpoint_data`):
- The stock checkpoint keeps the founders who live; the roster's state keeps
  the people born since (and which founders died).
- The restore is the stock `load_from_checkpoint`: it restores the founders,
  and the roster takes away the founders who died and rebuilds the born.
- A restored year plays on exactly as the live one.

This example relies on three core changes: a game master's `PlayerRoster`
(players join and leave a game; the asynchronous engine's threads follow),
entity agents adding and removing components, and bounded async logs.

### What a mind is asked

A person's acting component (the stock `ConcatActComponent`) builds one
`InteractiveDocument`. It holds each context component's text, then the
World's question as an exercise, then "Answer: " and the person's name for
the model to continue. This is a founder's first question in First
Farmers, as the model receives it (MOCK; customs shortened):

```
Who I am:
Kaka, born 29 years before the valley was settled.

Observations (ordered from oldest to latest):
[observation] Kaka is one of the first people to settle the valley.

A custom of ours (kinship) [c1]: We raise a child only when the stores can feed it.
A custom others expect of us (sharing) [c3]: We share our food with those near us who go hungry.
Something we know how to do (food) [c4]: Smoking fish over green alder. (level 2)
A habit of ours (food) [c7]: In lean years, gather before anything else.
...
:

Exercise: It is year 0. What does Kaka do this year? Choose one move:
- gather food
- rest
- move to better land
- ...
Answer with one of these moves, exactly as written, then the ids of the customs Kaka follows in doing it (or, to teach, the customs taught), if any.
Answer: Kaka gather food c4 c7
```

- **The order.** Identity and observations come first (the component's
  explicit order); the other components follow by key: each custom, then
  the Learner and `ObservationToMemory`, which say nothing. (The lone ":"
  is the stock `ObservationToMemory`, whose empty label and value still
  print a colon.)
- **The framing.** Concordia's models are framed to continue the user's
  sentences, so the answer begins with the name. Without it, live minds
  sowed a third as often (cycle 36).
- **What the World reads back.** The first move offered that the answer
  names (`world.first_move`, which also reads "Kaka sows a field"), and the
  ids of customs the GM once gave the person: those are enacted, and
  others may copy them (`cited` in the census).

## Tests

```bash
pytest examples/lifeways
```
