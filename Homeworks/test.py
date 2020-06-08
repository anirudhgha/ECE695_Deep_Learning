class car:
    def __init__(self, ev, num_wheels, brand):
        self.ev = ev
        self.num_wheels = num_wheels
        self.brand = brand
        self.battery = 0

    def __call__(self):
        return self.ev, self.num_wheels, self.brand

    def drive(self, distance):
        self.battery -= distance
        if self.battery < 0:
            print(
                "WHAT YOU DOING YOU FOOL YOU CAN'T DRIVE WITHOUT CHARGIN UR CAR"
            )

    def charge(self):
        self.battery = 100

    def get_battery(self):
        return self.battery


mycar = car(False, 4, "toyota")
yourcar = car(True, 11, "Tesla")

print(mycar())
print(yourcar())

yourcar.charge()
yourcar.drive(110)

print(yourcar.get_battery(), yourcar())