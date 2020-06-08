# HW 1

import random
import string

random.seed(0)


def generate_string(length):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(length))


class People:
    def __init__(self, name1, name2, name3, order):
        self.first_names = name1
        self.middle_names = name2
        self.last_names = name3
        self.order = order

    def __call__(self):
        a = sorted(self.last_names)
        for i in range(10):
            print(a[i])

    def __iter__(self):
        return PeopleIter(self)


class PeopleWithMoney(People):
    def __init__(self, persons):
        super().__init__(persons.first_names, persons.middle_names, persons.last_names, persons.order)
        self.wealth = [random.randint(0, 1000) for i in range(10)]

    def __call__(self):
        # sort the names based on the wealth
        final_first_names = [x for _, x in sorted(zip(self.wealth, self.first_names))]
        final_middle_names = [x for _, x in sorted(zip(self.wealth, self.middle_names))]
        final_last_names = [x for _, x in sorted(zip(self.wealth, self.last_names))]
        final_wealth = sorted(self.wealth)
        for ii in range(10):
            print(final_first_names[ii] + " " + final_middle_names[ii] + " " + final_last_names[ii] + " " + str(
                final_wealth[ii]))
        return

    def __iter__(self):
        return PeopleWithMoneyIter(self)


class PeopleIter:
    def __init__(self, persons):
        self.items1 = persons.first_names
        self.items2 = persons.middle_names
        self.items3 = persons.last_names
        self.order = persons.order
        self.index = -1

    def __iter__(self):
        return self

    def __next__(self):

        self.index += 1
        if self.index < len(self.items1) and self.index < len(self.items2) and self.index < len(self.items3):
            if self.order == "first_name_first":
                return self.items1[self.index] + " " + self.items2[self.index] + " " + self.items3[self.index]
            elif self.order == "last_name_first":
                return self.items3[self.index] + " " + self.items1[self.index] + " " + self.items2[self.index]
            elif self.order == "last_name_with_comma_first":
                return self.items3[self.index] + ", " + self.items1[self.index] + " " + self.items2[self.index]
        else:
            raise StopIteration

    next = __next__


class PeopleWithMoneyIter(PeopleIter):
    def __init__(self, personsWithMoney):
        self.items1 = personsWithMoney.first_names
        self.items2 = personsWithMoney.middle_names
        self.items3 = personsWithMoney.last_names
        self.wealth = personsWithMoney.wealth
        self.order = 'first_name_first'
        self.index = -1

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.items1) and self.index < len(self.items2) and self.index < len(self.items3):
            return super().__next__() + " " + str(self.wealth[self.index])
        else:
            return StopIteration
    next = __next__


name1, name2, name3 = "", "", ""
first_names, middle_names, last_names = [], [], []

for i in range(10):
    first_names.append(generate_string(5))
for i in range(10):
    middle_names.append(generate_string(5))
for i in range(10):
    last_names.append(generate_string(5))

"""
Task 1 - Task 5
"""
person1 = People(first_names, middle_names, last_names, "first_name_first")
iters1 = iter(person1)
for i in range(10):
    print(iters1.next())
print("\n")
person2 = People(first_names, middle_names, last_names, "last_name_first")
iters2 = iter(person2)
for i in range(10):
    print(iters2.next())
print("\n")
person3 = People(first_names, middle_names, last_names, "last_name_with_comma_first")
iters3 = iter(person3)
for i in range(10):
    print(iters3.next())
print("\n")

"""
Task 6
"""

person1()
print("\n")

"""
Task 7
"""
personWithMoney = PeopleWithMoney(person1)
iters4 = iter(personWithMoney)
print('\n')
for i in range(10):
    print(iters4.next())
print("\n")
personWithMoney()
