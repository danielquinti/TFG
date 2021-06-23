#!/usr/bin/python3
import itertools
#FILENAME="ACDC - Can_'t stand still.gp3"
#FILENAME="Harper, Ben - Alone (Live).gp4"
#FILENAME="Beatles (The) - All Together Now.gp3"
FILENAME="samples/Beatles (The) - All Together Now-tux.gp4"

class Reader:
        def __init__(self, filename):
                self.file=open(filename, "rb")
                self.CHUNKSIZE=4
                self.chunk=[]
                self.pos=self.CHUNKSIZE+1000
                self.globalPos=0

        def close(self):
                self.file.close()

        def _readChunk(self):
                self.chunk=self.file.read(self.CHUNKSIZE)
                if not self.chunk:
                        raise EOFError()
                self.pos=0
        def readBytes(self, nBytes):
                total=b''
                while nBytes>0:
                        if self.pos>=len(self.chunk):
                                self._readChunk()
                        nRead=nBytes
                        if (self.pos+nRead>len(self.chunk)):
                                nRead=len(self.chunk)-self.pos
                        total+=self.chunk[self.pos:self.pos+nRead]
                        self.pos+=nRead
                        self.globalPos+=nRead
                        nBytes-=nRead
                return total
        def readString(self):
                length=int.from_bytes(self.readBytes(1), byteorder="big")
                s=self.readBytes(length)
                return s.decode("utf8")

        def readInt(self):
                return int.from_bytes(self.readBytes(4), byteorder="little")
        def readByte(self):
                return self.readBytes(1)
        def skipBytes(self, n):
                self.readBytes(n)
        def printPos(self):
                print("Position -> %s"%hex(self.globalPos))

r=Reader(FILENAME)
#Version
version=r.readString()
print("VERSION:\t%s"%version)
r.readBytes(30-len(version))

#Information about the piece
int_size=r.readInt()
print("INT:\t%d"%int_size)
title=r.readString()
print("TITLE:\t%s"%title)
int_size=r.readInt()
print("INT:\t%d"%int_size)
subtitle=r.readString()
print("SUBTITLE:\t%s"%subtitle)
int_size=r.readInt()
print("INT:\t%d"%int_size)
interpreter=r.readString()
print("INTERPRETER:\t%s"%interpreter)
int_size=r.readInt()
print("INT:\t%d"%int_size)
album=r.readString()
print("ALBUM:\t%s"%album)
int_size=r.readInt()
print("INT:\t%d"%int_size)
copyright=r.readString()
print("COPYRIGHT:\t%s"%copyright)
int_size=r.readInt()
print("INT:\t%d"%int_size)
author=r.readString()
print("AUTHOR:\t%s"%author)
int_size=r.readInt()
print("INT:\t%d"%int_size)
tabauthor=r.readString()
print("TABAUTHOR:\t%s"%tabauthor)
int_size=r.readInt()
print("INT:\t%d"%int_size)
instructional=r.readString()
print("INSTRUCTIONAL:\t%s"%instructional)
r.printPos()
note=[]
note_lines=r.readInt()
for i in range(note_lines):
        int_size=r.readInt()
        print("INT:\t%d"%int_size)
        line=r.readString()
        note.append(line)
print("NOTE:\t%s"%note)
tripletFeel=int.from_bytes(r.readByte(), byteorder="big")!=0
print("TRIPLETFEEL:\t%s"%tripletFeel)

if "3.0" not in version:
        #Lyrics
        '''
        track=r.readInt()
        print("TRACK:\t%d"%track)
        lyrics=[]
        for i in range(3):
                chars=r.readInt()
                print("CHARS:\t%d"%chars)
                l=r.readBytes(chars).decode("utf8")
                print("LINE%d: %s"%(i,l))
                lyrics.append(l)
        print("LYRICS:\t%s"%lyrics)
        '''
        r.printPos()
        r.skipBytes(11*4)
        r.printPos()

#Other info
tempo=int.from_bytes(r.readByte(), byteorder="big")
print("TEMPO:\t%d"%tempo)
key=int.from_bytes(r.readByte(), byteorder="big")
print("KEY:\t%d"%key)
octave=int.from_bytes(r.readByte(), byteorder="big")
print("OCTAVE:\t%d"%octave)
midiChannels=[]
r.printPos()
for i in range(4):
        port=[]
        for j in range(16):
                instrument=r.readInt()
                volume=r.readByte()
                balance=r.readByte()
                chorus=r.readByte()
                reverb=r.readByte()
                phaser=r.readByte()
                tremolo=r.readByte()
                blank1=r.readByte()
                blank2=r.readByte()
                port.append((instrument,volume,balance,chorus,reverb,phaser,tremolo,blank1,blank2))
        midiChannels.append(port)
r.printPos()
#TODO??
r.skipBytes(6)
numMeasures=r.readInt()
print("NUMMEASURES:\t%d"%numMeasures)
numTracks=r.readInt()
print("NUMTRACKS:\t%d"%numTracks)
measures=[]
for i in range(numMeasures):
        header=r.readByte()[0]
        isDoubleBar=(header & 0b10000000) > 0
        hasTonality=(header & 0b1000000) > 0
        hasMarker=(header & 0b100000) > 0
        numberOfAlternate=(header & 0b10000) > 0
        endRepeat=(header & 0b1000) > 0
        beginRepeat=(header & 0b100) > 0
        hasDenominator=(header & 0b10) > 0
        hasNumerator=(header & 0b1) > 0

        if header!=0:
                print("\nMEASURE %d"%i)
                if isDoubleBar:
                        print("\tDOUBLE BAR")
                if beginRepeat:
                        print("\tSTART REPEAT")
                if hasNumerator:
                        numerator=int.from_bytes(r.readByte(), byteorder="big")
                        print("\tNUMERATOR:%d"%numerator)
                if hasDenominator:
                        denominator=int.from_bytes(r.readByte(), byteorder="big")
                        print("\tDENOMINATOR:%d"%denominator)
                if endRepeat:
                        numberOfRepeats=int.from_bytes(r.readByte(), byteorder="big")
                        print("\tNUMBER OF REPEATS:%d"%numberOfRepeats)
                if numberOfAlternate:
                        numberOfAlt=int.from_bytes(r.readByte(), byteorder="big")
                        print("\tNUMBER OF ALT:%d"%numberOfAlt)
                if hasMarker:
                        int_size=r.readInt()
                        #print("\tINT:\t%d"%int_size)
                        marker=r.readString()
                        red=r.readByte()[0]
                        green=r.readByte()[0]
                        blue=r.readByte()[0]
                        white=r.readByte()[0]

                        print("\tMARKER [rgb(%d,%d,%d)]:\t%s"%(red,green,blue,marker))
                if hasTonality:
                        tonality=int.from_bytes(r.readByte(), byteorder="big")
                        print("\tTONALITY:%d"%tonality)

r.printPos()
for i in range(numTracks):
        tHeader=r.readByte()[0]
        isBanjo=(tHeader & 0b100) > 0
        is12String=(tHeader & 0b10) > 0
        isDrums=(tHeader & 0b1) > 0

        print("\nTrack %d"%i)
        if isBanjo:
                print("\tBanjo")
        if is12String:
                print("\t12 String")
        if isDrums:
                print("\tDrums")
        name=r.readString()
        print("\tNAME: %s"%name)
        r.skipBytes(40-len(name))
        numStrings=r.readInt()
        print("\tNum. Strings: %d"%numStrings)
        tuning=[]
        for j in range(7):
                tuning.append(r.readInt())
        print("\tTuning: %s"%tuning)

        for n in ["Port","Channel","ChannelE","NumFrets","Capo"]:
                print("\t%s: %d"%(n,r.readInt()))

        red=r.readByte()[0]
        green=r.readByte()[0]
        blue=r.readByte()[0]
        white=r.readByte()[0]

        print("\tColor: rgb(%d,%d,%d)"%(red,green,blue))

r.printPos()
for i in range(numMeasures):
        for j in range(numTracks):
                numBeats=r.readInt()
                beats=[]
                for k in range(numBeats):
                        print("\nMeasure %d - Track %d - Beat %d/%d"%(i,j,k+1,numBeats))
                        r.printPos()
                        bHeader=r.readByte()[0]
                        hasStatus=(bHeader & 0b1000000) > 0
                        ntuplet=(bHeader & 0b100000) > 0
                        mixTableChange=(bHeader & 0b10000) > 0
                        hasEffects=(bHeader & 0b1000) > 0
                        hasText=(bHeader & 0b100) > 0
                        chordDiagram=(bHeader & 0b10) > 0
                        dotted=(bHeader & 0b1) > 0

                        if hasStatus:
                                status=r.readByte()
                                print("\tSTATUS: %d"%int.from_bytes(status, byteorder="big"))
                        duration=r.readByte()
                        print("\tDURATION: %d"%int.from_bytes(duration, byteorder="big"))
                        if ntuplet:
                                nTupletValue=r.readInt()
                                print("\tNTUPLET: %d"%nTupletValue)
                        if chordDiagram:
                                print("\tReading chord diagram at")
                                r.printPos()
                                cHeader=r.readByte()[0]
                                sharp=r.readByte()[0]
                                r.skipBytes(3)
                                root=r.readByte()[0]
                                cType=r.readByte()[0]
                                extra=r.readByte()[0]
                                bass=r.readInt()
                                dimaug=r.readInt()
                                add=r.readByte()[0]
                                name=r.readString()
                                r.skipBytes(20-len(name))
                                r.skipBytes(2)
                                fifth=r.readByte()[0]
                                ninth=r.readByte()[0]
                                eleventh=r.readByte()[0]
                                baseFret=r.readInt()
                                frets=[]
                                for l in range(7):
                                        frets.append(r.readInt())
                                barres=r.readByte()[0]
                                barreList=[]
                                for l in range(5):
                                        barreList.append(r.readByte()[0])
                                barreStarts=[]
                                for l in range(5):
                                        barreStarts.append(r.readByte()[0])
                                barreEnds=[]
                                for l in range(5):
                                        barreEnds.append(r.readByte()[0])
                                ommissions=[]
                                for l in range(7):
                                        ommission.append(r.readByte()[0])
                                r.skipBytes(1)
                                fingering=[]
                                for l in range(7):
                                        fingering.append(r.readByte()[0])
                                showFingering=r.readByte()[0]

                        if hasText:
                                int_size=r.readInt()
                                print("INT:\t%d"%int_size)
                                text=r.readString()
                                print("\tTEXT: %s"%text)
                        if hasEffects:
                                print("\tReading chord diagram at")
                                r.printPos()
                                eHeader=r.readByte()[0]
                                hasStroke=(eHeader & 0b1000000) > 0
                                hasTapping=(eHeader & 0b100000) > 0
                                eHeader2=r.readByte()[0]
                                hasTremolo=(eHeader2 & 0b100) > 0
                                hasPickstroke=(eHeader2 & 0b10) > 0
                                hasRasguedo=(eHeader2 & 0b1) > 0

                                if hasTapping:
                                        tapping=r.readByte()[0]

                                if hasTremolo:
                                        bType=r.readByte()[0]
                                        bHeight=r.readInt()
                                        numPoints=r.readInt()
                                        points=[]
                                        for p in range(numPoints):
                                                time=r.readInt()
                                                vPos=r.readInt()
                                                vibrato=r.readByte()[0]
                                                points.append((time,vPos,vibrato))

                                if hasStroke:
                                        upStroke=r.readByte()[0]
                                        downStroke=r.readByte()[0]

                                if hasRasguedo:
                                        rasguedo=r.readByte()[0]

                                if hasPickstroke:
                                        pickStroke=r.readByte()[0]

                        if mixTableChange:
                                print("\tReading mix table change at")
                                r.printPos()
                                instrument=r.readByte()[0]
                                volume=r.readByte()[0]
                                pan=r.readByte()[0]
                                chorus=r.readByte()[0]
                                reverb=r.readByte()[0]
                                phaser=r.readByte()[0]
                                tremolo=r.readByte()[0]
                                tempo=r.readInt()
                                if volume!=0xff:
                                        volChangeD=r.readByte()[0]
                                if pan!=0xff:
                                        panChangeD=r.readByte()[0]
                                if chorus!=0xff:
                                        chorusChangeD=r.readByte()[0]
                                if reverb!=0xff:
                                        reverbChangeD=r.readByte()[0]
                                if phaser!=0xff:
                                        phaserChangeD=r.readByte()[0]
                                if tremolo!=0xff:
                                        tremoloChangeD=r.readByte()[0]
                                if tempo!=0xff:
                                        tempoChangeD=r.readByte()[0]
                                allTracks=r.readByte()[0]

                        print("Reading strings")
                        r.printPos()
                        involvedStrings=r.readByte()[0]
                        usesSixth=(involvedStrings & 0b1000000) > 0
                        usesFifth=(involvedStrings & 0b100000) > 0
                        usesFourth=(involvedStrings & 0b10000) > 0
                        usesThird=(involvedStrings & 0b1000) > 0
                        usesSecond=(involvedStrings & 0b100) > 0
                        usesFirst=(involvedStrings & 0b10) > 0
                        print(involvedStrings)

                        notes=[]
                        for pos,active in enumerate([usesSixth,usesFifth,usesFourth,usesThird,usesSecond,usesFirst]):
                                if active:
                                        print("Reading string #%d"%pos)
                                        #NOTE
                                        nHeader=r.readByte()[0]
                                        right=(nHeader & 0b10000000) > 0
                                        accentuated=(nHeader & 0b1000000) > 0
                                        hasNoteType=(nHeader & 0b100000) > 0
                                        hasNoteDynamic=(nHeader & 0b10000) > 0
                                        hasEffects=(nHeader & 0b1000) > 0
                                        ghost=(nHeader & 0b100) > 0
                                        dotted=(nHeader & 0b10) > 0
                                        hasDuration=(nHeader & 0b1) > 0
                                        print(nHeader)


                                        if hasNoteType:
                                                noteType=r.readByte()[0]
                                                #r.skipBytes(1)

                                        if hasDuration:
                                                duration=r.readByte()[0]
                                                ntuplet=r.readByte()[0]

                                        strength=6
                                        if hasNoteDynamic:
                                                strength=r.readByte()[0]

                                        fret=0
                                        if hasNoteType:
                                                fret=r.readByte()[0]

                                        if right:
                                                fingeringL=r.readByte()[0]
                                                fingeringR=r.readByte()[0]

                                        if hasEffects:
                                                neHeader=r.readByte()[0]
                                                hasGraceNote=(neHeader & 0b10000) > 0
                                                letRing=(neHeader & 0b1000) > 0
                                                hasSlideFrom=(neHeader & 0b100) > 0
                                                hasHammerFrom=(neHeader & 0b10) > 0
                                                hasBend=(neHeader & 0b1) > 0

                                                neHeader2=r.readByte()[0]
                                                leftHandVibrato=(neHeader2 & 0b1000000) > 0
                                                trill=(neHeader2 & 0b100000) > 0
                                                harmonic=(neHeader2 & 0b10000) > 0
                                                hasSlideFrom2=(neHeader2 & 0b1000) > 0
                                                tremoloPicking=(neHeader2 & 0b100) > 0
                                                palmMute=(neHeader2 & 0b10) > 0
                                                stacatto=(neHeader2 & 0b1) > 0

                                                if hasBend:
                                                        bType=r.readByte()[0]
                                                        bHeight=r.readInt()
                                                        numPoints=r.readInt()
                                                        points=[]
                                                        for p in range(numPoints):
                                                                time=r.readInt()
                                                                vPos=r.readInt()
                                                                vibrato=r.readByte()[0]
                                                                points.append((time,vPos,vibrato))

                                                if hasGraceNote:
                                                        fret=r.readByte()[0]
                                                        dynamic=r.readByte()[0]
                                                        transition=r.readByte()[0]
                                                        duration=r.readByte()[0]

                                                if tremoloPicking:
                                                        tremoloPickingValue=r.readByte()[0]

                                                if hasSlideFrom2:
                                                        slide=r.readByte()[0]

                                                if harmonic:
                                                        harmonicValue=r.readByte()[0]

                                                if trill:
                                                        trillFret=r.readByte()[0]
                                                        trillPeriod=r.readByte()[0]
                                        notes.append((6-pos, fret, strength))
                                        r.printPos()
                        beats.append((k,notes))
                if j==3:
                        print(beats)

r.close()
