import matplotlib.colors as colors

# This file is intended to parse a Teams file, and provide the relevant info
# List of valid colors. No spaces, all lower case
#  https://matplotlib.org/stable/gallery/color/named_colors.html

# Choose the font color based on 
#  https://stackoverflow.com/questions/3942878/how-to-decide-font-color-in-white-or-black-depending-on-background-color

# if (red*0.299 + green*0.587 + blue*0.114) > 149 use #000000 else use #ffffff
class Team:
    def __init__(self, team_file):
        if team_file is None:
            return None
        with open(team_file, 'r') as f:
            team_name = f.readline().strip()
            team_color = f.readline().strip()
            members = list()
            for line in f.readlines():
                members.append(line.strip())
            self.name = team_name
            self.colorname = team_color
            self.members = members
            hex_bg_color = colors.cnames[self.colorname]
            print(hex_bg_color)
            red, green, blue = colors.to_rgb(hex_bg_color)
            # OpenCV expects BGR, not RGB :(
            self.bg_color = (blue*255, green*255, red*255)
            if (red*255*0.299 + green*255*0.587 + blue*255*0.114) > 149:
                # Use Black font
                self.font_color = (0, 0, 0)
                self.accent_color = (255, 255, 255)
            else:
                # Use White font
                self.font_color = (255, 255, 255)
                self.accent_color = (0, 0, 0)
            print(self.font_color)

    def __repr__(self):
        return f"{self.name} ({self.colorname}) {len(self.members)} players"



if __name__ == "__main__":
    team_file = "team_one.txt"

    t = Team(team_file=team_file)
    print(t)