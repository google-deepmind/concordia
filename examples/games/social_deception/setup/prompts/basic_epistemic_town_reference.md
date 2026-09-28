# Basic Epistemic Town: Master Role Reference

This document provides a comprehensive reference for all 22 roles in the *Basic
Epistemic Town* script, as implemented in the Social Deception simulation
environment.

--------------------------------------------------------------------------------

## 🟦 Townsfolk (Good)

### Witness

-   **Game Master Summary**: Learns that 1 of 2 players is a particular
    Townsfolk.
-   **Player Introduction**: You start knowing that 1 of 2 players is a
    particular Townsfolk. During the first night, you learn that a specific
    Townsfolk role is held by one of two named players.
-   **Night Action (Night 1)**: The Game Master reveals that a specific
    Townsfolk is held by one of two players.

### Researcher

-   **Game Master Summary**: Learns that 1 of 2 players is a particular Outsider
    (or that zero are in play).
-   **Player Introduction**: You start knowing that 1 of 2 players is a
    particular Outsider (or that zero are in play). On the first night, you
    learn one of two players holds an Outsider role, or learn that none exist.
-   **Night Action (Night 1)**: The Game Master reveals that an Outsider is held
    by one of two players, or shows '0'.

### Investigator

-   **Game Master Summary**: Learns that 1 of 2 players is a particular Minion.
-   **Player Introduction**: You start knowing that 1 of 2 players is a
    particular Minion. On the first night, you learn that a specific Minion role
    is held by one of two named players.
-   **Night Action (Night 1)**: The Game Master reveals that a specific Minion
    is held by one of two players.

### Matchmaker

-   **Game Master Summary**: Learns how many pairs of adjacent evil players
    exist in the circle.
-   **Player Introduction**: You start knowing how many pairs of adjacent evil
    players are seated next to each other. Two evil players seated adjacently
    count as one pair; three adjacent evil players count as two pairs.
-   **Night Action (Night 1)**: You learn the count of adjacent evil player
    pairs.

### Empath

-   **Game Master Summary**: Learns how many of their 2 living neighbors are
    Evil.
-   **Player Introduction**: Each night, you learn how many of your 2 living
    neighbors are evil (0, 1, or 2). Dead players are skipped when determining
    neighbors.
-   **Night Action (Every Night)**: How many of your two living neighbors are
    Evil?

### Seer

-   **Game Master Summary**: Chooses 2 players each night to learn if either is
    a Demon.
-   **Player Introduction**: Each night, choose 2 players: you learn if either
    is a Demon (Yes/No). One Good player acts as a decoy and always registers as
    a Demon to you.
-   **Night Action (Every Night)**: Pick two players. Are either of them the
    Demon?

### Gravedigger

-   **Game Master Summary**: Learns the true role of the player executed today.
-   **Player Introduction**: Each night after day 1, you learn the true role of
    the player who died by execution during the day.
-   **Night Action (Night 2+)**: Learn the role of the player executed today.

### Guardian

-   **Game Master Summary**: Protects one player from the Demon's night attack.
-   **Player Introduction**: Each night after day 1, choose a living player
    other than yourself: they are protected from being killed by the Demon
    tonight.
-   **Night Action (Night 2+)**: Choose a player (not yourself) to protect.

### Specter

-   **Game Master Summary**: If killed at night, wakes to learn one player's
    character.
-   **Player Introduction**: If you die during the night, you are awakened
    immediately to choose one player and learn their true character role.
-   **Night Action (Night of Death)**: Pick one player to learn their character.

### Innocent

-   **Game Master Summary**: If nominated by a Townsfolk, the nominator is
    executed immediately.
-   **Player Introduction**: The first time you are nominated, if the nominator
    is a Townsfolk, they are executed immediately and the day ends.
-   **Night Action**: Passive / None.

### Executioner

-   **Game Master Summary**: Once per game during day, publicly shoots a player:
    kills if Demon.
-   **Player Introduction**: Once per game during the day, you may declare a
    shot against a player: if that player is the Demon, they die immediately.
-   **Night Action**: Passive / None.

### Soldier

-   **Game Master Summary**: Immune to Demon night attacks.
-   **Player Introduction**: You cannot be killed by the Demon's night attack.
-   **Night Action**: Passive / None.

### Mayor

-   **Game Master Summary**: Peace victory if 3 alive with no execution; night
    attack bounce.
-   **Player Introduction**: If only 3 players live and no execution occurs,
    your team wins. If attacked at night, another player might die instead.
-   **Night Action**: Passive / None.

--------------------------------------------------------------------------------

## 🟦 Outsiders (Good)

### Servant

-   **Game Master Summary**: Can only vote if their chosen master votes.
-   **Player Introduction**: Each night, choose a master: tomorrow, you may only
    vote if your master votes too.
-   **Night Action (Every Night)**: Choose your master for tomorrow.

### Drunk

-   **Game Master Summary**: Thinks they are a Townsfolk; ability yields
    false/impaired info.
-   **Player Introduction**: You think you are a Townsfolk, but you are the
    Drunk and have no real ability.
-   **Night Action**: Passive / Receives impaired info.

### Outcast

-   **Game Master Summary**: May register as Evil, Minion, or Demon.
-   **Player Introduction**: You are Good, but you might register as Evil and as
    a Minion or Demon to information abilities.
-   **Night Action**: Passive.

### Saint

-   **Game Master Summary**: If executed during the day, Evil wins immediately.
-   **Player Introduction**: If you die by execution during the day, the game
    ends immediately and Evil wins.
-   **Night Action**: Passive.

--------------------------------------------------------------------------------

## 🟥 Minions (Evil)

### Poisoner

-   **Game Master Summary**: Poisons one player each night to impair their
    ability.
-   **Player Introduction**: Each night, choose a player to poison: their
    ability fails or gives false information through tomorrow dusk.
-   **Night Action (Every Night)**: Choose a player to poison.

### Spy

-   **Game Master Summary**: Sees the complete game state; may register as Good.
-   **Player Introduction**: Each night, you see the complete GameTracker state.
    You may register as Good and as a Townsfolk/Outsider.
-   **Night Action (Every Night)**: View the true game state.

### Apprentice

-   **Game Master Summary**: Becomes the Demon if the Demon dies with 5+ alive.
-   **Player Introduction**: If the Demon dies and 5 or more players remain
    alive, you become the Demon.
-   **Night Action**: Passive.

### Corruptor

-   **Game Master Summary**: Setup modifier: +2 Outsiders, -2 Townsfolk.
-   **Player Introduction**: There are 2 extra Outsiders in play (setup
    modifier).
-   **Night Action**: Passive.

--------------------------------------------------------------------------------

## 🟥 Demon (Evil)

### Demon

-   **Game Master Summary**: Kills one player each night; can pass the mantle
    via self-kill.
-   **Player Introduction**: Night 1: Learn safe bluffs. Night 2+: Choose a
    player to kill. If you target yourself, a living Minion becomes the Demon.
-   **Night Action (Every Night)**: Choose a target to kill.
