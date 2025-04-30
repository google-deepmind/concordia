from concordia.associative_memory import character_sheet

player_config = character_sheet.AgentConfig(
  name = 'Whiskeyjack',
  gender = 'male',
  class_and_level = 'Fighter Level 1',
  race = 'human',
  experience_points = 'Gain experience at Milestones',
  background = 'Soldier',
  alignment = 'Lawful Good',
)

print(player_config)
